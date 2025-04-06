from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import math

class CustomConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride=1, padding=0):
        # 保存参数供反向传播使用
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding

        # 手动实现卷积
        output = torch.conv2d(
            input, weight, bias, 
            stride=stride, padding=padding
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        # 计算梯度
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.conv_transpose2d(
                grad_output, weight, None, 
                stride=stride, padding=padding
            )
        if ctx.needs_input_grad[1]:
            grad_weight = torch.conv2d(
                input.transpose(0, 1), grad_output.transpose(0, 1), 
                None, stride=stride, padding=padding
            ).transpose(0, 1)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None
    
class CustomConv2dModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 初始化参数
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size
        ))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return CustomConv2dFunction.apply(
            x, self.weight, self.bias, 
            self.stride, self.padding
        )

class Net(nn.Module):
    def __init__(self, linear_type='native'):
        super(Net, self).__init__()
        if linear_type == 'native':
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
        if linear_type == 'function':
            self.conv1 = CustomConv2dModule(1, 32, 3, 1)
            self.conv2 = CustomConv2dModule(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

  
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch, prof):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        prof.step()  # 记录每一步

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')



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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # 添加 --linear-type 参数
    parser.add_argument('--linear-type', type=str, default='native',
                        choices=['native', 'module', 'function'],
                        help='Linear layer type (native/module/function)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=3)  # 减少 epoch 数以快速测试性能
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-model', action='store_true', default=False)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # 数据加载
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # 根据 --linear-type 参数初始化模型
    model = Net(linear_type=args.linear_type).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # 训练与性能分析
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        with_stack=True
    )
    prof.start()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, prof)
        test(model, device, test_loader)
        scheduler.step()
    prof.stop()
    # 打印性能报告
    if prof is not None:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    # 打印性能摘要
    print(torch.profiler.ProfilerActivity.CPU)


if __name__ == '__main__':
    main()