import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class Layer3CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(Layer3CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.act=nn.ReLU
        self.mlp1 = nn.Linear(input_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
       
    def forward(self, x):
        x1 = self.mlp1(x)
        x1 = self.act(x1)
        x1 = self.dropout(x1)# Dropout
        
        
        x2 = self.mlp2(x1)
        x2 = self.dropout(x2)# Dropout
        
        x3 = self.mlp3(x2)
        return x3

class BU_CNN(nn.Module):
    def __init__(self, pooling_type='max', in_channels=3, init_channels=32):
        super(BU_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self._get_pooling(pooling_type, kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(init_channels, init_channels*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self._get_pooling(pooling_type, kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(init_channels*2, init_channels*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self._get_pooling(pooling_type, kernel_size=2, stride=2)
        )
    
        # Additional pooling layer to get to 16x16
        self.pool4 = self._get_pooling(pooling_type, kernel_size=2, stride=2)
    
    def _get_pooling(self, pooling_type, **kwargs):
        if pooling_type.lower() == 'mean':
            return nn.AvgPool2d(**kwargs)
        elif pooling_type.lower() == 'max':
            return nn.MaxPool2d(**kwargs)
        else:
            raise ValueError("Pooling type must be 'mean' or 'max'")
    
    def forward(self, x):
        x = self.conv1(x)  # [B, C, 256, 256] → [B, 32, 128, 128]
        x = self.conv2(x)  # [B, 32, 128, 128] → [B, 64, 64, 64]
        x = self.conv3(x)  # [B, 64, 64, 64] → [B, 128, 32, 32]
        x = self.pool4(x)  # [B, 128, 32, 32] → [B, 128, 16, 16]
        return x
    
class MLP1(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16*16, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim))
    
    def forward(self, x):
        return self.net(x)    

class MLP2(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16*16, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim))
    
    def forward(self, x):
        return self.net(x)

class MLP3(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16*16, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim))
    
    def forward(self, x):
        return self.net(x)

class FeatureFusionSystem(nn.Module):
    """整合三个MLP和特征融合的完整系统"""
    def __init__(self, embed_dim=32):
        super().__init__()
        # 初始化三个MLP
        self.mlp1 = MLP1(embed_dim)
        self.mlp2 = MLP2(embed_dim)
        self.mlp3 = MLP3(embed_dim)
        
        # 特征融合模块
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 32 * 32)
        )
    
    def forward(self, x1, x2, x3):
        """
        输入:
            x1, x2, x3: 三个16x16的输入特征 [batch, 256] 或 [batch, 16, 16]
        输出:
            融合后的32x32特征 [batch, 32, 32]
        """
        # 分别通过三个MLP
        feat1 = self.mlp1(x1)  # [batch, embed_dim]
        feat2 = self.mlp2(x2)  # [batch, embed_dim]
        feat3 = self.mlp3(x3)  # [batch, embed_dim]
        
        # 特征融合
        fused = torch.cat([feat1, feat2, feat3], dim=-1)  # [batch, embed_dim*3]
        return self.fusion(fused).view(-1,1, 32, 32)  # [batch, 32, 32]
    
class Decoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            # 32x32 → 64x64
            nn.ConvTranspose2d(in_channels, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64x64 → 128x128
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 128x128 → 256x256
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 最终输出层
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class CompleteModel(nn.Module):
    """端到端完整模型"""
    def __init__(self, embed_dim=32):
        super().__init__()
        self.feature_fusion = FeatureFusionSystem(embed_dim)
        self.decoder = Decoder()
    
    def forward(self, x1, x2, x3):
        features = self.feature_fusion(x1, x2, x3)  # [b,1,32,32]
        return self.decoder(features)  # [b,3,256,256]

def parse_args():
    parser = argparse.ArgumentParser(description='Bottom-Up CNN Configuration')
    
    # Model parameters
    parser.add_argument('--pooling', type=str, default='max', 
                       choices=['mean', 'max'], help="Pooling type: 'mean' or 'max'")
    parser.add_argument('--in_channels', type=int, default=3,
                       help='Number of input channels (default: 3 for RGB)')
    parser.add_argument('--init_channels', type=int, default=32,
                       help='Number of initial channels (default: 32)')
    
    # Input parameters
    parser.add_argument('--input_size', type=int, default=256,
                       help='Input image size (default: 256)')
    
    # Training control
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for testing (default: 1)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Initialize model with command-line arguments
    model = BU_CNN(
        pooling_type=args.pooling,
        in_channels=args.in_channels,
        init_channels=args.init_channels
    )
    
    # Print model summary
    print(f"\nModel Configuration:")
    print(f"- Pooling type: {args.pooling}")
    print(f"- Input channels: {args.in_channels}")
    print(f"- Initial channels: {args.init_channels}")
    print(f"- Input size: {args.input_size}x{args.input_size}")
    
    # Create dummy input based on arguments
    dummy_input = torch.randn(
        args.batch_size, 
        args.in_channels, 
        args.input_size, 
        args.input_size
    )
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}\n")
    
