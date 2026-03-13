import torch
import torch.nn as nn

# 最简单的自编码器
class SimpleAE(nn.Module):
    def __init__(self, input_dim=100, latent_dim=20):
        super().__init__()
        # 编码器：压缩数据
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # 解码器：重建数据
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)      # 压缩到隐空间
        x_recon = self.decoder(z) # 重建
        return x_recon, z

# 训练代码
def train_ae(model, dataloader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        for batch in dataloader:
            x = batch  # 假设batch就是输入数据
            
            # 前向传播
            x_recon, z = model(x)
            
            # 计算重建误差
            loss = loss_fn(x_recon, x)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')