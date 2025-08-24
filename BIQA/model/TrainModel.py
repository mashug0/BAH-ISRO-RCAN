import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
import pandas as pd

def compute_sharpness_batch(patches, sharpness_func):
    sharpness_scores = []
    for i in range(patches.shape[0]):
        img = patches[i, 0].cpu().numpy()
        img_uint8 = (img * 255).astype(np.uint8)
        score = sharpness_func(img_uint8)
        sharpness_scores.append(score)
    return torch.tensor(sharpness_scores, dtype=torch.float32)

import torch
import time
from tqdm import tqdm


def train_model(model, dataloader , val_dataset , glcm_func, sharpness_func, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    print(f"\n{'Epoch':<7} | {'Loss':<10} | {'MAE':<8} | Time")
    print("-" * 40)

    best_mae = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_mae = 0
        start_time = time.time()

        model.train()
        for patches, mos in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            # patches: [B, P, 1, 32, 32]
            B, P, C, H, W = patches.shape

            patches = patches.view(B * P, C, H, W).to(device, non_blocking=True)
            mos = mos.to(device, non_blocking=True).unsqueeze(1)

            glcm = glcm_func(patches).to(device, non_blocking=True)          # [B*P, 6]
            sharpness = compute_sharpness_batch(patches , sharpness_func=sharpness_func).unsqueeze(1).to(device, non_blocking=True)  # [B*P, 1]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                features = model(patches, glcm, sharpness)                   # [B*P, feature_dim=100]
                features = features.view(B, P, -1)                            # [B, P, 100]
                pooled = model.pool_moments_batches(features)                 # [B, 200]
                pred_mos = model.regressor(pooled)                            # [B, 1]

                loss = criterion(pred_mos, mos)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            mae = torch.abs(pred_mos - mos).mean().item()

            epoch_loss += loss.item()
            epoch_mae += mae

        avg_loss = epoch_loss / len(dataloader)
        avg_mae = epoch_mae / len(dataloader)
        elapsed = time.time() - start_time
        print(f"{epoch+1:<7} | {avg_loss:<10.4f} | {avg_mae:<8.4f} | {elapsed:.1f}s")

        if avg_mae < best_mae:
            best_mae = avg_mae
            torch.save(model.state_dict(), "best_model.pth")

        
    print("Training Complete")

    
