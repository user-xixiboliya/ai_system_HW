import torch
import numpy as np
import os
import gzip
from torchvision import transforms
import torchvision
import cv2
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torch.profiler import profile,ProfilerActivity,tensorboard_trace_handler
import matplotlib.pyplot as plt

#########
#dataset#
#########

class MyMNISTDataset(Dataset):
    def __init__(self, folder_root, data_name,label_name,transform,target_transform=None):
        self.transform = transform
        self.data_name = data_name
        self.label_name = label_name
        self.target_transform = target_transform
        self.folder_root = folder_root
        self.images,self.labels = self.load_dataset()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # numpy数组 [28, 28]
        label = self.labels[idx]
        image = Image.fromarray(image, mode='L')  # 'L'灰度图
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def load_dataset(self):
        with gzip.open(os.path.join(self.folder_root, self.label_name), 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(os.path.join(self.folder_root, self.data_name), 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            images = images.reshape(len(labels), 28, 28)  # [样本数, 28, 28]

        return images, labels

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = MyMNISTDataset(
    folder_root='../data/MNIST/raw',  # 假设原始gz文件在此目录下
    data_name='train-images-idx3-ubyte.gz',
    label_name='train-labels-idx1-ubyte.gz',
    transform=transform
)
test_dataset = MyMNISTDataset(
    folder_root='../data/MNIST/raw',
    data_name='t10k-images-idx3-ubyte.gz',
    label_name='t10k-labels-idx1-ubyte.gz',
    transform=transform
)
#########
# model #
#########

import torch.nn as nn
import torch.nn.functional as F  # 确保添加了这行

class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.channel = channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channel,
                      out_channels=channel,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),  # (1,28,28)
            nn.BatchNorm2d(32),  # (32,24,24)
            nn.ReLU(),
            nn.MaxPool2d(2)  # (32,12,12)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5),  # (16,8,8)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  # (16,4,4)
        )
        self.reslayer1 = ResidualBlock(32)
        self.reslayer2 = ResidualBlock(16)
        self.fc = nn.Linear(256, 10)  # 这里的输入256是因为16*4*4=256

    def forward(self, x):
        out = self.conv1(x) # 32，12，12
        out = self.reslayer1(out)
        out = self.conv2(out) #16，4，4
        out = self.reslayer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

###########
#  utils  #
###########
from torch.utils.tensorboard import SummaryWriter
def test (model):
    dummy_input = torch.randn(64, 1, 28, 28)  # 批量大小64
    output = model(dummy_input)
    writer = SummaryWriter('logs/mynet_mnist_experiment_1')
    writer.add_graph(model, dummy_input)
    print(output.shape)

################
## parameters ##
################
config = {
    'num_epoch': 15,
    'batch_size': 512,
    'lr': 1e-3
}


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    writer = SummaryWriter('logs/mynet_mnist_experiment_1')

    prof = torch.profiler.profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    ###############
    # Dataloader  #
    ###############

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    total_step = 0
    with prof:
        for epoch in range(config['num_epoch']):
            model.train()
            total_loss = 0

            for batch_idx , (images, labels) in enumerate(train_loader):
                data, target = images.to(device), labels.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if batch_idx % 50 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{config["num_epoch"]}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')

                prof.step()
                total_step += 1
            # Validation
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()

            avg_loss = total_loss / len(train_loader)
            test_loss /= len(test_loader)
            accuracy = 100. * correct / len(test_dataset)

            print(f'Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
            writer.add_scalar('Train Loss', avg_loss, epoch)
            writer.add_scalar('Test Loss', test_loss, epoch)
            writer.add_scalar('Accuracy', accuracy, epoch)

        writer.close()
        print("Training complete!")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        if not os.path.exists("trace.json"):
            prof.export_chrome_trace("trace.json")