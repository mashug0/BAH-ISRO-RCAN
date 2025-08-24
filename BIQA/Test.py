import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
import cv2

class CreateTestDataset(Dataset):
    def __init__(self, csv_path, image_dir, reference_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.reference_dir = reference_dir
        self.transform = transform
        self.data = []

        for idx, row in self.df.iterrows():
            degraded_filename = f"{row['filename']}.tif"
            degraded_path = os.path.join(image_dir, degraded_filename)

            # Reconstruct reference filename by removing level + score
            parts = row['filename'].split("_")
            reference_name = f"{parts[0]}_{0}_{1.0000}.tif"  # assumes format: name_level_score
            reference_path = os.path.join(reference_dir, reference_name)

            if not os.path.exists(degraded_path) or not os.path.exists(reference_path):
                continue

            self.data.append((degraded_path, reference_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        degraded_path, reference_path = self.data[idx]

        degraded = cv2.imread(degraded_path, cv2.IMREAD_GRAYSCALE)
        reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

        # Normalize and convert to tensors
        degraded_tensor = torch.tensor(degraded, dtype=torch.float32).unsqueeze(0) / 255.0
        reference_tensor = torch.tensor(reference, dtype=torch.float32).unsqueeze(0) / 255.0

        return degraded_tensor, reference_tensor

