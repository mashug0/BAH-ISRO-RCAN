import torch
import torch.nn as nn
import torch.optim as adam

class BIECON(nn.Module):
    def __init__(self):
        super(BIECON, self).__init__()
        # Step 1: Feature Extraction (Conv1 to Conv4)
        self.features = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, padding=1),  # 32x32 -> 32x32
            nn.ELU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 100), # 100D feature vector
            nn.ReLU() 
        )
        
        # Step 2: Regression (MLP for MOS prediction)
        self.regressor = nn.Sequential(
            nn.Linear(200, 64),  # Input dim: 100 (mean) + 100 (std)
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output MOS in [0, 1]
        )

    def forward(self, x):
        # Feature extraction for all patches
        x = self.features(x)  # Output: [batch, 100]
        return x

    def pool_moments(self, patch_features):
        # Input: [num_patches, 100]
        mean = torch.mean(patch_features, dim=0)  # μ (100-D)
        std = torch.std(patch_features, dim=0)    # σ (100-D)
        return torch.cat([mean, std], dim=0)      # [200-D]

# Training Loop (Simplified)
def train_biecon(model, dataloader):
    optimizer = adam.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(100):
        for patches, mos in dataloader:
            # Step 1: Extract patch features
            features = model(patches)  # [batch, 100]
            
            # Step 2: Pooling (mean + std)
            pooled = model.pool_moments(features)  # [200]
            
            # Regression to MOS
            pred_mos = model.regressor(pooled.unsqueeze(0))
            
            # Loss and backprop
            loss = criterion(pred_mos, mos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

