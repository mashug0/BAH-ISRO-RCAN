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
degrador.save_hr_patches()
for i in range(10):
    degrador.get_LR1()
degrador.get_LR2()

dataset = CreateDataset(
    HR_dir="Data\\Spacenet\\HR",
    LR1_dir="Data\\Spacenet\\LR1",
    LR2_dir="Data\\Spacenet\\LR2"
)

loader = DataLoader(dataset=dataset , batch_size=8 , shuffle=True)

for batch in loader:
    hr, lr1, lr2 = batch['HR'], batch['LR1'], batch['LR2']
    print(hr.shape, lr1.shape, lr2.shape)
    break