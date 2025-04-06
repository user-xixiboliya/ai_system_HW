# assignment1

实验要求：
```
Ubuntu 18.04 LTS x86_64 以上
python 3.10.0
torch 1.12.0
```

新建虚拟环境:
```bash
conda create -n <your_env_name> python=3.10
conda activate <your-env>
pip install opencv-python
pip install tensorboard
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
需要自行修改修改`mynet.py`下的MNIST文件夹的路径。
```python
train_dataset = MyMNISTDataset(
    folder_root='../data/MNIST/raw',  ## 修改
    data_name='train-images-idx3-ubyte.gz',
    label_name='train-labels-idx1-ubyte.gz',
    transform=transform
)
test_dataset = MyMNISTDataset(
    folder_root='../data/MNIST/raw',  ## 修改
    data_name='t10k-images-idx3-ubyte.gz',
    label_name='t10k-labels-idx1-ubyte.gz',
    transform=transform
)
```
# 项目结构
```bash
.
├── README.md
├── download_MNIST.py
├── logs
│ ├── mynet_mnist_experiment_1
│ └── profiler
├── mnist_basic.py
├── mnist_profiler.py
├── mnist_tensorboard.py
└── mynet.py
```
# setup
运行:
```bash
python mynet.py
```
可以先退出本ssh会话，再通过`ssh -L 6006:localhost:6006 username@remote_server_ip -p yourport`进行会话设置，也可以使用另一个终端进行ssh链接，并进行端口转发设置

使用tensorboard进行转发， 本代码的log存储在logs下：

```bash
tensorboard --logdir=logs --port=6006
```
