import argparse
import logging
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms

logger = logging.getLogger('mnist_enhanced')
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


class EnhancedNet(nn.Module):
    def __init__(self, hidden_size):
        super(EnhancedNet, self).__init__()
        # 卷积层增强
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)  # 保持28x28尺寸
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 全连接层增强
        self.fc1 = nn.Linear(128 * 2 * 2, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 10)

    def forward(self, x):
        # 特征提取
        x = F.relu(self.bn1(self.conv1(x)))  # 32x28x28
        x = F.max_pool2d(x, 2)  # 32x14x14

        x = F.relu(self.bn2(self.conv2(x)))  # 64x10x10
        x = F.max_pool2d(x, 2)  # 64x5x5

        x = F.relu(self.bn3(self.conv3(x)))  # 128x5x5
        x = F.max_pool2d(x, 2)  # 128x2x2

        # 分类器
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
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
        if batch_idx % 100 == 0:
            logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                        f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(model, device, test_loader):
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

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    logger.info(f'\nTest set: Average loss: {test_loss:.4f}, '
                f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    return accuracy


def main(args):
    device = torch.device(f'cuda:{args["device"]}'
                          if args['device'] != -1 and torch.cuda.is_available()
                          else 'cpu')

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

    train_loader = data.DataLoader(
        datasets.MNIST(args['data_dir'], train=True, download=True, transform=train_transform),
        batch_size=args['batch_size'], shuffle=True, num_workers=2)

    test_loader = data.DataLoader(
        datasets.MNIST(args['data_dir'], train=False, transform=test_transform),
        batch_size=1000, shuffle=False)

    model = EnhancedNet(hidden_size=1024).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test_acc = test(model, device, test_loader)
        logger.debug('Test accuracy: %.2f%%', test_acc)

    logger.info('Final test accuracy: %.2f%%', test_acc)

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--data_dir", type=str,
                        default='../data', help="data directory")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--batch_num", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N',
                        help='hidden layer size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--device', type=int, default=0,
                        help='selected device for training -1:cpu')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    start_time = time.time()
    params = get_params().__dict__
    main(params)

    import time


    def get_params():
        # 这里简单返回一个示例对象，你需要根据实际情况修改
        class Params:
            pass

        params = Params()
        return params


    def main(params):
        # 这里是 main 函数的具体逻辑，你需要根据实际情况修改
        print("Running main function with params:", params)


    if __name__ == '__main__':
        start_time = time.time()
        params = get_params().__dict__
        main(params)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"代码运行时间: {elapsed_time} 秒")
