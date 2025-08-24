import torch
import torch.nn as nn
from torchvision import transforms

class SGLCMAwareBIECON(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_features = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 100),
            nn.ReLU()
        )
        self.fusion = nn.Linear(100 + 6 + 1, 100)
        self.regressor = nn.Sequential(
            nn.Linear(200, 64),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, glcm_features=None, sharpness_score=None):
        cnn_feats = self.cnn_features(x)                      # Likely [B, C, H, W]
        cnn_feats = cnn_feats.view(cnn_feats.size(0), -1)     # Flatten to [B, N]

    # Make sure glcm_features is 2D
        if glcm_features is not None and glcm_features.dim() == 3:
            glcm_features = glcm_features.view(glcm_features.size(0), -1)

    # Make sure sharpness_score is 2D: [B, 1]
        if sharpness_score is not None:
            if sharpness_score.dim() == 1:
              sharpness_score = sharpness_score.unsqueeze(1)
            elif sharpness_score.dim() == 3:
                sharpness_score = sharpness_score.view(sharpness_score.size(0), -1)

        if glcm_features is not None and sharpness_score is not None:
            fused = torch.cat([cnn_feats, glcm_features, sharpness_score], dim=1)
            fused = self.fusion(fused)
        else:
            fused = cnn_feats

        return fused
    def pool_moments(self, patch_features):
        # Input: [num_patches, 100]
        mean = torch.mean(patch_features, dim=0)  # μ (100-D)
        std = torch.std(patch_features, dim=0)    # σ (100-D)
        return torch.cat([mean, std], dim=0)     
     
    def pool_moments_batches(self , patch_features):
            mean = torch.mean(patch_features, dim=1)  # [B, 100]
            std = torch.std(patch_features, dim=1)    # [B, 100]
            pooled = torch.cat([mean, std], dim=1)    # [B, 200]
            return pooled


