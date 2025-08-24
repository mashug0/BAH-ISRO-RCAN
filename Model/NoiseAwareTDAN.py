import torch
import torch.nn as nn

from Model.Allignment import TDANAlignment
from Model.FeatureExtractor import FeatureExtractor
from Model.NoiseEstimator import NoiseEstimator

class NoiseAwareTDAN(nn.Module):
    def __init__(self, feature_channels=32):
        super().__init__()
        self.noise_estimator = NoiseEstimator()
        self.feature_extractor = FeatureExtractor(in_channels=2, out_channels=feature_channels)
        self.align = TDANAlignment(in_channels=feature_channels, channels=feature_channels)
        self.fusion = nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=1)

    def forward(self, ref, support, return_losses=True):
        noise_ref = self.noise_estimator(ref)
        noise_supp = self.noise_estimator(support)
        ref_in = torch.cat([ref, noise_ref], dim=1)
        support_in = torch.cat([support, noise_supp], dim=1)
        ref_feature = self.feature_extractor(ref_in)
        support_feature = self.feature_extractor(support_in)
        aligned_features, alignment_loss = self.align(ref_feature, support_feature, True)
        fused_feat = self.fusion(torch.cat([ref_feature, aligned_features], dim=1))
        if return_losses:
            return fused_feat, alignment_loss
        return fused_feat
