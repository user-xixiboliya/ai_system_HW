# BSD 3-Clause License

# Copyright (c) 2017, 
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file has been changed for education and teaching purpose

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.onnx.symbolic_opset9 import convolution
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn.modules.utils import _pair
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # 添加TensorBoard支持

#import our c++ module
import mylinear_cpp 
import myconv_cpp

class myLinearFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        #output = input.mm(weight.t())
        output = mylinear_cpp.forward(input, weight)
        return output[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        #grad_input = grad_weight = None
        #grad_input = grad_output.mm(weight)
        #grad_weight = grad_output.t().mm(input)
        grad_input, grad_weight = mylinear_cpp.backward(grad_output, input, weight)
        return grad_input, grad_weight

class myLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(myLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, input):
        return myLinearFunction.apply(input, self.weight)

##################
#self convolution#
##################
class myconvolution_pool_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight,stride=1,padding=1):
        ctx.save_for_backward(input, weight)
        #output = input.mm(weight.t())
        input,pool_out, conv_out, indices = myconv_cpp.forward(input, weight,stride,padding)
        ctx.save_for_backward(input, weight, conv_out,indices)
        ctx.stride = stride
        ctx.padding = padding
        return pool_out

    @staticmethod
    def backward(ctx, grad_pool):
        input, weight, conv_out, indices = ctx.saved_tensors
        grad_input, grad_weight = myconv_cpp.backward(
            grad_pool,
            conv_out,
            indices,
            input,
            weight,
            ctx.stride,
            ctx.padding)
        return grad_input, grad_weight,None,None


class myConvolution_pool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0):
        super(myConvolution_pool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))

    def forward(self,input):
        return myconvolution_pool_function.apply(input,self.weight,self.stride,self.padding)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = myConvolution_pool(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 10)
        self.fc2 = myLinear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        x = F.relu(x)
        # x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch,writer):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        if batch_idx % args.log_interval == 0:
            writer.add_scalar('Train/Loss', loss.item(),
                            epoch * len(train_loader) + batch_idx)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    writer.add_scalar('Train/Epoch Loss', total_loss / len(train_loader), epoch)


def test(model, device, test_loader,epoch,writer):
    model.eval()
    test_loss = 0
    correct = 0
    progress_bar = tqdm(test_loader, desc='Testing', leave=False)

    with torch.no_grad():
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            progress_bar.set_postfix({
                'acc': f'{100. * correct / len(test_loader.dataset):.1f}%'
            })
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    writer.add_scalar('Test/Loss', test_loss, epoch)
    writer.add_scalar('Test/Accuracy', accuracy, epoch)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    writer = SummaryWriter(comment=f'_LR{args.lr}_BS{args.batch_size}')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
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

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch,writer)
        test(model, device, test_loader,epoch,writer)
        scheduler.step()

        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()