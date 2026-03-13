import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor，并归一化到[0,1]
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1,1]，适配GAN
])

# 下载训练集（第一次运行会自动下载）
train_dataset = datasets.MNIST(
    root='./data',  # 保存路径
    train=True,     # 训练集
    transform=transform,
    download=True   # 自动下载
)

# 下载测试集
test_dataset = datasets.MNIST(
    root='./data',
    train=False,    # 测试集
    transform=transform,
    download=True
)

# 创建DataLoader
train_loader = DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=True
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=64, 
    shuffle=False
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"图像形状: {train_dataset[0][0].shape}")  # torch.Size([1, 28, 28])