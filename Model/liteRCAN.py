import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.NoiseAwareTDAN import NoiseAwareTDAN

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


class LiteDualRCAN(nn.Module):
  def __init__(self , in_channels = 1, n_features = 32 , n_rg = 5 , n_rcab = 3 , scale = 2):
    super(LiteDualRCAN , self).__init__()
    
    self.tdan = NoiseAwareTDAN(feature_channels=n_features)

    self.body = nn.Sequential(*[ResidualBlock(n_features, n_rcab) for _ in range(n_rg)])
    self.tail = nn.Sequential(
      nn.Conv2d(n_features , n_features * (scale**2) , kernel_size = 3 , padding = 1),
      nn.PixelShuffle(scale),
      nn.Conv2d(n_features , in_channels , kernel_size = 3, padding =1)
    )

  def forward(self , lr1 , lr2):
    fused , alligment_loss = self.tdan(lr1 , lr2, True)
    res = self.body(fused)
    out = self.tail(res + fused)
    return out ,alligment_loss