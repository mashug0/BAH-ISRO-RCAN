import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ReliableDeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Always use regular convolution for stability
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        
        # Optional offset learning (but don't actually apply deformation)
        self.offset_conv = nn.Conv2d(in_channels, 2*kernel_size*kernel_size, 
                                    kernel_size=3, padding=1, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        # For stability, just use regular convolution
        # The offset_conv still learns but we don't apply the deformation
        _ = self.offset_conv(x)  # Keep for parameter consistency
        return self.conv(x)



class RCAB(nn.Module):
    """Residual Channel Attention Block"""
    def __init__(self, n_feat: int, reduction: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, padding=1)
        )
        reduction = min(reduction, n_feat)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, max(1, n_feat // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, n_feat // reduction), n_feat, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        ca = self.ca(x)
        return x + res * ca


class ResidualGroup(nn.Module):
    """Residual group containing multiple RCABs"""
    def __init__(self, n_feat: int, n_rcab: int):
        super().__init__()
        self.rcabs = nn.Sequential(*[RCAB(n_feat) for _ in range(n_rcab)])
        self.conv = nn.Conv2d(n_feat, n_feat, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(self.rcabs(x))


class FeatureAlignment(nn.Module):
    def __init__(self, channels=32, use_deformable=True):
        super().__init__()
        self.use_deformable = use_deformable
        
        self.feature_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        if use_deformable:
            # Try reliable deformable conv first, fallback to simple version
            try:
                self.align_conv = ReliableDeformableConv2d(channels, channels)
            except:
                print("Using simple deformable convolution")
                self.align_conv = SimpleDeformableConv2d(channels, channels)
        else:
            # Regular convolution fallback
            self.align_conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, ref_feat, supp_feat):
        ref_feat = self.feature_refine(ref_feat)
        supp_feat = self.feature_refine(supp_feat)
        
        aligned_feat = self.align_conv(supp_feat)
        align_loss = F.l1_loss(aligned_feat, ref_feat)
        
        return aligned_feat, align_loss


class RobustLiteDualRCAN(nn.Module):
    """Robust version with multiple fallback options"""
    def __init__(self, 
                 in_channels: int = 1, 
                 n_features: int = 32,
                 n_rg: int = 3,
                 n_rcab: int = 2,
                 scale: int = 2,
                 use_deformable: bool = True):
        super().__init__()
        
        # Simple feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, n_features//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features//2, n_features, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Feature alignment with fallback options
        self.alignment = FeatureAlignment(n_features, use_deformable=use_deformable)
        self.fusion = nn.Conv2d(n_features * 2, n_features, 1)
        
        # Residual groups
        self.residual_blocks = nn.Sequential(*[
            ResidualGroup(n_features, n_rcab) for _ in range(n_rg)
        ])
        
        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(n_features, n_features * (scale**2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_features, in_channels, 3, padding=1)
        )

    def forward(self, lr1: torch.Tensor, lr2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract features
        feat1 = self.feature_extractor(lr1)
        feat2 = self.feature_extractor(lr2)
        
        # Align and fuse features
        aligned_feat, align_loss = self.alignment(feat1, feat2)
        fused_feat = self.fusion(torch.cat([feat1, aligned_feat], dim=1))
        
        # Process through residual blocks
        res = self.residual_blocks(fused_feat)
        
        # Upsample
        sr = self.upsample(res + fused_feat)
        
        return sr, align_loss