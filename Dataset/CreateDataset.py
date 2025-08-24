''''''

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class CreateDataset(Dataset):
    def __init__(self , HR_dir , LR1_dir , LR2_dir , transform = None):
        self.HR_dir = HR_dir
        self.LR1_dir = LR1_dir
        self.LR2_dir = LR2_dir
        self.transform = transform if transform else transforms.ToTensor()
        
        self.hr_files = sorted(
            [
                f for f in os.listdir(self.HR_dir)
                if f.endswith('.tif')
            ]
        )
        
        self.lr1_files = [f.replace('.tif', '_0.tif') for f in self.hr_files]
        self.lr2_files = [f.replace('.tif', '_1.tif') for f in self.hr_files]

    def __len__(self):
        return len(self.hr_files)
    def __getitem__(self, idx):
        hr_path = os.path.join(self.HR_dir, self.hr_files[idx])
        lr1_path = os.path.join(self.LR1_dir, self.lr1_files[idx])
        lr2_path = os.path.join(self.LR2_dir, self.lr2_files[idx])

        hr = Image.open(hr_path).convert('L')
        lr1 = Image.open(lr1_path).convert('L')
        lr2 = Image.open(lr2_path).convert('L')

        hr = self.transform(hr)  # [1, H, W]
        lr1 = self.transform(lr1)
        lr2 = self.transform(lr2)

        return {
            'HR': hr,
            'LR1': lr1,
            'LR2': lr2,
            'name': os.path.splitext(self.hr_files[idx])[0]
        }