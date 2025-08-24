import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.DeformConv2d import DeformConv2d

class TDANAlignment(nn.Module):
    def __init__(self, in_channels=32, channels=32):
        super().__init__()
        self.pre_deform = nn.Conv2d(in_channels, channels, 3, padding=1)     # 32 → 32
        self.bottleneck = nn.Conv2d(channels * 2, channels, 3, padding=1)    # 64 → 32
        self.offset_conv = nn.Conv2d(channels, 18, 3, padding=1)             # offsets
        self.deform_conv = DeformConv2d(channels, channels, 3, padding=1)    # 32 → 32

    def forward(self, ref_feature, supp_feature, alignment_loss=False):
        ref_feature = self.pre_deform(ref_feature)       # [B, 32, H, W]
        supp_feature = self.pre_deform(supp_feature)     # [B, 32, H, W]
        concat = torch.cat([ref_feature, supp_feature], dim=1)  # [B, 64, H, W]
        bottleneck = self.bottleneck(concat)             # [B, 32, H, W]
        offset = self.offset_conv(bottleneck)            # [B, 18, H, W]
        aligned_feat = self.deform_conv(supp_feature, offset)  # [B, 32, H, W]
        if alignment_loss:
            return aligned_feat, F.l1_loss(aligned_feat, ref_feature)
        return aligned_feat
