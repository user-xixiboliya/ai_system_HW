import argparse
import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import torch.multiprocessing as mp

logger = logging.getLogger('mnist_ddp')
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.INFO)


def ddp_worker(rank, world_size, args):
    # 使用 env:// 已经设置好 MASTER_ADDR/PORT
    dist.init_process_group(
        backend=args['backend'],
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    main(rank, world_size, args)

class EnhancedNet(nn.Module):
    def __init__(self, hidden_size):
        super(EnhancedNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 2 * 2, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch, rank):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args['batch_num'] and batch_idx >= args['batch_num']:
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if rank == 0 and batch_idx % 100 == 0:
            samples = batch_idx * args['batch_size'] * args['world_size']
            total = len(train_loader.dataset)
            percent = 100. * batch_idx / len(train_loader)
            logger.info(f'Train Epoch: {epoch} [{samples}/{total} ({percent:.0f}%)] Loss: {loss.item():.6f}')


def test(model, device, test_loader, rank, world_size):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 使用Distributed Tensor进行结果聚合
    test_loss_tensor = torch.tensor(test_loss, device=device)
    correct_tensor = torch.tensor(correct, device=device)
    dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = test_loss_tensor.item() / len(test_loader.dataset)
        accuracy = 100. * correct_tensor.item() / len(test_loader.dataset)
        logger.info(f'Test set: Average loss: {test_loss:.4f}, '
                    f'Accuracy: {correct_tensor.item()}/{len(test_loader.dataset)} '
                    f'({accuracy:.2f}%)')
        return accuracy
    return None



def main(rank, world_size, args):
    # 初始化进程组
    dist.init_process_group(
        backend=args['backend'],
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

    # 设置设备
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    args['world_size'] = world_size

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 数据集加载
    train_dataset = datasets.MNIST(
        args['data_dir'],
        train=True,
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.MNIST(
        args['data_dir'],
        train=False,
        transform=test_transform
    )

    # 分布式采样器
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True
    )

    # 模型初始化
    model = EnhancedNet(args['hidden_size']).to(device)
    model = DDP(model, device_ids=[rank])

    # 优化器
    optimizer = optim.SGD(model.parameters(),
                          lr=args['lr'],
                          momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=5,
                                          gamma=0.1)

    # 训练循环
    start_time = time.time()
    best_acc = 0.0
    for epoch in range(1, args['epochs'] + 1):
        train_sampler.set_epoch(epoch)
        train(args, model, device, train_loader, optimizer, epoch, rank)
        scheduler.step()
        current_acc = test(model, device, test_loader, rank, world_size)

        if rank == 0 and current_acc and current_acc > best_acc:
            best_acc = current_acc

    if rank == 0:
        total_time = time.time() - start_time
        logger.info(f'Training completed in {total_time:.2f} seconds')
        logger.info(f'Best test accuracy: {best_acc:.2f}%')


def get_params():
    parser = argparse.ArgumentParser(description='DDP MNIST Training')
    # 添加num_gpus参数
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_num', type=int, default=None)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--backend', type=str,
                        default='nccl',
                        choices=['nccl', 'gloo', 'mpi'])
    return parser.parse_args()

if __name__ == '__main__':
    args = get_params()
    args_dict = vars(args)
    backend = args_dict['backend']

    if backend == 'mpi':
        # 1) 用 mpirun/mpiexec 启动，不要在脚本里用 spawn
        #    mpirun -n 4 python train.py --backend mpi
        dist.init_process_group(backend='mpi')
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        main(rank, world_size, args_dict)

    else:
        # 2) NCCL/Gloo，用 torch.multiprocessing.spawn
        available = torch.cuda.device_count()
        num_procs = args_dict.get('num_gpus') or available
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        mp.spawn(
            ddp_worker,  # 不再用 lambda
            args=(num_procs, args_dict),  # world_size, 参数字典
            nprocs=num_procs,
            join=True
        )