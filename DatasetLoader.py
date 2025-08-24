from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
import cv2
from Dataset.Degrador import Degrador
from Dataset.CreateDataset import CreateDataset
import os
from torch.utils.data import DataLoader

degrador = Degrador()
# degrador.save_hr_patches()
# while len(os.listdir("Data\\Spacenet\\LR1")) < len(os.listdir("Data\\Spacenet\\HR")):
# degrador.save_16bit_patches(max_workers=os.cpu_count(), read_path="bah_2025\\train_lr" , path="Data\\BAH\\LR" , title="HR")

dataset = CreateDataset(
    HR_dir="Data\\BAH\\HR",
    LR1_dir="Data\\BAH\\LR",
    LR2_dir="Data\\BAH\\LR"
)

loader = DataLoader(dataset=dataset , batch_size=8 , shuffle=True)
print(len(loader))
for batch in loader:
    hr, lr1, lr2 = batch['HR'], batch['LR1'], batch['LR2']
    print(hr.shape, lr1.shape, lr2.shape)
    break