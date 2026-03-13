import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ae import SimpleAE, train_ae
import matplotlib.pyplot as plt
import os

# 设置随机种子
torch.manual_seed(42)

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
print("加载MNIST数据集...")
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=False  # 已经下载了
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=False
)

# 创建DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 获取图像维度
sample_img, _ = train_dataset[0]
img_shape = sample_img.shape
print(f"图像形状: {img_shape}")
print(f"输入维度: {img_shape.numel()}")

# 创建模型（调整输入维度以适应MNIST）
input_dim = img_shape.numel()  # 28*28*1 = 784
latent_dim = 32
model = SimpleAE(input_dim=input_dim, latent_dim=latent_dim)

print(f"\n模型结构:")
print(f"输入维度: {input_dim}")
print(f"隐层维度: {latent_dim}")
print(model)

# 训练模型
print("\n开始训练...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = model.to(device)

# 修改训练函数以适配MNIST
def train_ae(model, dataloader, epochs=50, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)
            x = x.view(x.size(0), -1)  # 展平图像: [batch, 1, 28, 28] -> [batch, 784]

            # 前向传播
            x_recon, z = model(x)

            # 计算重建误差
            loss = loss_fn(x_recon, x)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.6f}')

    return model

# 训练模型
trained_model = train_ae(model, train_loader, epochs=50, device=device)

# 保存模型
model_path = 'ae_model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': input_dim,
    'latent_dim': latent_dim
}, model_path)
print(f"\n模型已保存到: {model_path}")

# 测试模型 - 可视化重建效果
print("\n可视化重建效果...")
model.eval()
with torch.no_grad():
    # 获取测试样本
    test_images, _ = next(iter(test_loader))
    test_images = test_images[:8].to(device)  # 取8张图
    test_images_flat = test_images.view(test_images.size(0), -1)

    # 重建
    recon_images, _ = model(test_images_flat)

    # 恢复形状
    recon_images = recon_images.view(-1, 1, 28, 28)
    test_images = test_images.view(-1, 1, 28, 28)

    # 反归一化到 [0, 1]
    test_images = (test_images + 1) / 2
    recon_images = (recon_images + 1) / 2

    # 转为numpy
    test_images = test_images.cpu().numpy()
    recon_images = recon_images.cpu().numpy()

    # 可视化
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        # 原图
        axes[0, i].imshow(test_images[i, 0], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)

        # 重建图
        axes[1, i].imshow(recon_images[i, 0], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12)

    plt.tight_layout()
    plt.savefig('reconstruction_results.png', dpi=150, bbox_inches='tight')
    print("重建结果已保存到: reconstruction_results.png")
    plt.show()

print("\n训练完成！")
