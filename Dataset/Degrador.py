'''Functions To Generate The LR images'''

import os
import io
import re
import math
import random
import threading
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image , to_tensor
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm

class Degrador:    
    
    #Basic Setup for Degradation
    def __init__(self,
                base_path = "spacenet-pan\\SN6_buildings_AOI_11_Rotterdam_train\\train\\AOI_11_Rotterdam\\PAN",
                HR_path = "Data\\Spacenet\\HR",
                LR1_path = "Data\\Spacenet\\LR1",
                LR2_path = "Data\\Spacenet\\LR2"):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.HR_target_size = (128,128)
        self.LR_target_size = (64,64)
        
        self.HR_path = HR_path
        self.LR1_path = LR1_path
        self.LR2_path = LR2_path
        
        self.base_path = base_path
        
        self.file_info_list = []
        try:
            for self.foldername, self.subfolders, self.filenames in os.walk(self.base_path):
                for filename in self.filenames:
                    filepath = os.path.join(self.foldername, filename)
                    file_size = os.path.getsize(filepath)
                    self.file_info_list.append({
                        'name': filename,
                        'path': filepath,
                        'size': file_size
                        })
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    
    def downsample(self, img):
    # Convert to tensor if not already
        if not isinstance(img, torch.Tensor):
            try:
                tensor = to_tensor(img)  # Converts to CxHxW and [0,1] range
            except Exception as e:
                raise ValueError(f"Failed to convert image to tensor: {e}")
        
        # Add batch dimension if needed
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        
        # Perform downsampling
        try:
            downsampled = F.interpolate(tensor,
                                    scale_factor=0.5,
                                    mode='bicubic',
                                    align_corners=False,
                                    antialias=True)
            return downsampled.squeeze(0)  # Remove batch dimension
        except Exception as e:
            raise RuntimeError(f"Downsampling failed: {e}")

    def add_noise(self, img, std=0.09):
        """Add Gaussian noise to image"""
        # Ensure we have a tensor
        if not isinstance(img, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")
        
        # Add batch dimension if needed
        if img.ndim == 3:
            img = img.unsqueeze(0)
        
        # Generate and add noise
        try:
            noise = torch.randn_like(img) * std
            img_noisy = img + noise
            img_noisy = torch.clamp(img_noisy, 0.0, 1.0)
            return img_noisy.squeeze(0)  # Remove batch dimension
        except Exception as e:
            raise RuntimeError(f"Noise addition failed: {e}")
    
    def add_pixel_shift(self , img):
        shift_x = np.random.uniform(0.3 , 0.7)
        shift_y = np.random.uniform(0.3 , 0.7)
        
        rows,cols = img.shape[:2]
        M = np.float32([[1,0,shift_x],
                        [0,1,shift_y]])
        
        return cv2.warpAffine(img , M , (cols , rows) , flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def lr2_from_lr1(self, fileinfo):
        try:
            filepath = fileinfo['path']
            filename = fileinfo['name']
            
            img = cv2.imread(filepath , cv2.IMREAD_UNCHANGED)
            
            shifted = self.add_pixel_shift(img)
            
            filename_base = os.path.splitext(filename)[0].replace('_lr1' , '')
            save_path = os.path.join(self.LR2_path , f"{filename_base}_lr2.tif")
            cv2.imwrite(save_path , shifted)
        except Exception as e:
            return f"❌ Error on {fileinfo['name']}: {e}"
    
    
    def get_LR1(self, max_workers = os.cpu_count()):
        file_info_list = []
        
        for root, _, files in os.walk(self.HR_path):
            for fname in files:
                if fname.endswith(".tif") or fname.endswith(".png"):
                    file_info_list.append({
                        'name': fname,
                        'path': os.path.join(root, fname)
                    })
        
        os.makedirs(self.LR1_path , exist_ok=True)
        lock = threading.Lock()
        
        def process_and_save(fileinfo):
            try:
                filepath = fileinfo['path']
                filename = fileinfo['name']
                
                try:
                    img = Image.open(filepath)
                except Exception as e:
                    return f"Error on {fileinfo['name']}: {e}" 
                
                img_ds = self.downsample(img=img)
                img_ds = self.add_noise(img_ds, std=0.01)
                
                img_pill = to_pil_image(img_ds)
                filename_base = os.path.splitext(filename)[0]
                save_path = os.path.join(self.LR1_path, f"{filename_base}_lr1.tif")

                img_pill.save(save_path)
                
                return f"Done:{filename}"
            except Exception as e:
                return f"Error on {fileinfo['name']}: {e}"  
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_and_save , fi) for fi in file_info_list]
            with tqdm(total=len(futures), desc="Parallel Processing", unit="img") as pbar:
                for f in as_completed(futures):
                    _ = f.result()  # Can log if needed
                    pbar.update(1)   
            
    
    def get_LR2(self , max_workers = os.cpu_count()):
        os.makedirs(self.LR2_path , exist_ok=True)
        
        file_info_list = []
        
        for root, _, files in os.walk(self.LR1_path):
            for fname in files:
                if fname.endswith(".tif") or fname.endswith(".png"):
                    file_info_list.append({
                        'name': fname,
                        'path': os.path.join(root, fname)
                    })
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.lr2_from_lr1, fileinfo)
                for fileinfo in file_info_list
            ]
            with tqdm(total=len(futures), desc="Generating LR2", unit="img") as pbar:
                for f in as_completed(futures):
                    _ = f.result()  # Optional: collect or log
                    pbar.update(1)
        
    def extract_hr_patches(self ,img_np, patch_size = 512 , stride = 256 ):
        H, W = img_np.shape
        patches = []
        for top in range(0, H - patch_size + 1, stride):
            for left in range(0, W - patch_size + 1, stride):
                patch = img_np[top:top+patch_size, left:left+patch_size]
                patches.append(patch)
        return patches

    def save_16bit_patches(self, max_workers , read_path,path , title):
        os.makedirs(path , exist_ok=True)
        
        file_info_list = []
        for foldername , _ , filenames in os.walk(read_path):
            for filename in filenames:
                filepath = os.path.join(foldername , filename)
            
                file_info_list.append(
                    {
                        'name':filename,
                        'path':filepath
                    }
                )
        
        lock = threading.Lock()

        def process_and_save(fileinfo):
            
            filepath = fileinfo['path']
            filename = fileinfo['name']
                
                
            img = cv2.imread(filepath , cv2.IMREAD_UNCHANGED) 
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img_8bit = img_norm.astype(np.uint8)
            
            save_path = os.path.join(path, filename)
            cv2.imwrite(save_path , img_8bit)
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_and_save , fi) for fi in file_info_list]
            with tqdm(total=len(futures), desc="Saving Extractions", unit="img") as pbar:
                for f in as_completed(futures):
                    _ = f.result()  # Can log if needed
                    pbar.update(1)
        
    def save_hr_patches(self , patch_size=512 , stride =256 , max_workers = os.cpu_count()):
        os.makedirs(self.HR_path , exist_ok=True)
        lock = threading.Lock()
        
        def process_and_save(fileinfo):
            try:
                filepath = fileinfo['path']
                filename = fileinfo['name']
                
                try:
                    img = cv2.imread(filepath , cv2.IMREAD_UNCHANGED) 
                except Exception as e:
                    return f"Error on {fileinfo['name']}: {e}" 
                
                img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img_8bit = img_norm.astype(np.uint8)
    
                patches = self.extract_hr_patches(img_np=img_8bit , patch_size=patch_size , stride=stride)
                for i,patch in enumerate(patches):
                    # part = filename[:-4]
                    # img_name = part[61:]
                    name = os.path.splitext(filename)[0]
                    try:
                        aoi_match = re.search(r'AOI_(\d+)', name)
                        tile_match = re.search(r'tile_(\d+)', name)
                    except IndexError:
                        aoi = "AOIxx"
                        tile_id = "tileXXXX"
                    aoi_id = f"AOI{aoi_match.group(1)}" if aoi_match else "AOIxx"
                    tile_id = f"tile{tile_match.group(1)}" if tile_match else "tilexxxx"
                    
                    save_path = os.path.join(self.HR_path , f"{aoi_id}_{tile_id}_{i}.tif")
                    # print(filename)
                    cv2.imwrite(save_path , patch)
                
                return f"Done:{filename}"
            except Exception as e:
                return f"Error on {fileinfo['name']}: {e}" 
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_and_save , fi) for fi in self.file_info_list]
            with tqdm(total=len(futures), desc="Parallel Processing", unit="img") as pbar:
                for f in as_completed(futures):
                    _ = f.result()  # Can log if needed
                    pbar.update(1)
