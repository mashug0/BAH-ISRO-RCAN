import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os

def split_into_patches(image, patch_size=32):
    h, w = image.shape[:2]
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return np.array(patches)

class CreateDataset(Dataset):
    def __init__(self, csv_path, image_dir, patchsize=32, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.patch_size = patchsize
        
        # Store patches grouped by index/image
        self.grouped_patches = []
        self.mos_scores = []
        
        for idx, row in self.df.iterrows():
            img_path = os.path.join(image_dir, f"{row['filename']}.tif")
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Could not read image: {img_path}")
                continue  # Skip if image can't be read

            patches = split_into_patches(image, patch_size=self.patch_size)  # [num_patches, H, W]
            self.grouped_patches.append(patches)
            self.mos_scores.append(row['mos_score'])

    def __len__(self):
        return len(self.grouped_patches)

    def __getitem__(self, index):
        patches = self.grouped_patches[index]
        # Convert to tensor [num_patches, 1, H, W] and normalize
        patch_tensor = torch.tensor(patches, dtype=torch.float32).unsqueeze(1) / 255.0
        mos = torch.tensor(self.mos_scores[index], dtype=torch.float32)
        return patch_tensor, mos
