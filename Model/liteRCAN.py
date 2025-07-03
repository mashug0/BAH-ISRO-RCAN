import torch
import torch.nn as nn
import torch.nn.functional as F

class RCAB(nn.Module):
  def __init__(self, n_feat, reduction=16):
    super(RCAB, self).__init__()
    self.body = nn.Sequential(
        nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)
    )
    self.channel_attention = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),  # Reduces spatial dims to 1x1
        nn.Conv2d(n_feat, n_feat // reduction, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(n_feat // reduction, n_feat, kernel_size=1),
        nn.Sigmoid()
    )
  def forward(self , x):
    res = self.body(x)
    ca = self.channel_attention(x)
    res = res*ca
    return res + x

class ResidualBlock(nn.Module):
  def __init__(self , n_features , n_rcab):
    super(ResidualBlock , self).__init__()
    modules = [RCAB(n_features) for _ in range (n_rcab)]
    modules.append(nn.Conv2d(n_features , n_features , kernel_size = 3 , padding =1))
    self.body = nn.Sequential(*modules)
  
  def forward(self,x):
    return self.body(x)+x

class DualInputFusion(nn.Module):
  def __init__(self , n_features):
    super(DualInputFusion , self).__init__()
    
    self.fusion = nn.Sequential(
      nn.Conv2d(n_features * 2 , n_features , kernel_size = 1),
      # nn.ReLU(inplace = True)
      nn.ReLU(inplace=True)
    )
  def forward(self , x1 , x2):
    x = torch.cat([x1 , x2],dim=1)
    return x

class LiteDualRCAN(nn.Module):
  def __init__(self , in_channels = 1, n_features = 32 , n_rg = 5 , n_rcab = 3 , scale = 2):
    super(LiteDualRCAN , self).__init__()
    self.head = nn.Conv2d(in_channels , n_features , kernel_size = 3 , padding = 1)
    self.head2 = nn.Conv2d(in_channels , n_features , kernel_size = 3 , padding = 1)

    self.dual_input_fusion = nn.Sequential(
            nn.Conv2d(n_features * 2, n_features, kernel_size=1),  # Channel reduction
            nn.ReLU(inplace=True)
          )

    self.body = nn.Sequential(*[ResidualBlock(n_features, n_rcab) for _ in range(n_rg)])
    self.tail = nn.Sequential(
      nn.Conv2d(n_features , n_features * (scale**2) , kernel_size = 3 , padding = 1),
      nn.PixelShuffle(scale),
      nn.Conv2d(n_features , in_channels , kernel_size = 3, padding =1)
    )

  def forward(self , lr1 , lr2):
    f1 = self.head(lr1)
    f2 = self.head2(lr2)

    fused = self.dual_input_fusion(torch.cat([f1 , f2] , dim=1))
    res = self.body(fused)
    out = self.tail(res + fused)
    return out