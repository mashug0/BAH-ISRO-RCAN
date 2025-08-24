import os
import io
import cv2
import math
import random
import threading
import csv
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm
from Degrador import compute_mos
class ImageDegrador:
    
    def __init__(self,  
                 downsample_factor_range=(2, 4),
                 target_size=(128, 128),
                 base_path="/content/ClearVision/data/Scrapped", 
                 corruption_path="/content/ClearVision/data/Corrupted"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Changed kernel sizes to match paper: 7x7 to 21x21 (odd only)
        self.kernel_sizes = list(range(7, 22, 2))
        
        self.downsample_factor_range = downsample_factor_range
        self.target_size = target_size
        self.base_path = base_path
        self.corruption_path = corruption_path
        
        # Fixed jitter parameter (was 0.05, now 0.5 as requested)
        self.jitter_shift = 0.5
        
        # Automatic degradation distribution - NO NOISE as per final request
        self.auto_degradation_config = {
            'blur_intensity_range': (0.1, 1.0),      # Random blur strength
            'jitter_intensity_range': (0.1, 1.0),    # Random jitter strength
            'jpeg_quality_range': (30, 95),          # Random JPEG quality
            'num_operations_range': (2, 4),          # Random number of operations
            'operation_probability': {               # Probability of each operation
                'iso_blur': 0.8,
                'aniso_blur': 0.7, 
                'jpeg_compression': 0.6,
                'spatial_jitter': 0.5
            }
        }
        
        self.file_info_list = []
        try:
            for foldername, subfolders, filenames in os.walk(self.base_path):
                for filename in filenames:
                    filepath = os.path.join(foldername, filename)
                    file_size = os.path.getsize(filepath)
                    self.file_info_list.append({
                        'name': filename,
                        'path': filepath,
                        'size': file_size
                    })
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    
    def get_sigma_ranges(self, scale_factor):
        """Get sigma ranges based on scale factor as per paper"""
        if scale_factor == 2:
            iso_sigma = (0.1, 2.4)
            aniso_sigma = (0.5, 6)
        else:  # scale_factor == 4
            iso_sigma = (0.1, 2.8)
            aniso_sigma = (0.5, 8)
        return iso_sigma, aniso_sigma
    
    def get_random_parameters(self):
        """Generate random degradation parameters automatically"""
        config = self.auto_degradation_config
        
        params = {
            'blur_intensity': random.uniform(*config['blur_intensity_range']),
            'jitter_intensity': random.uniform(*config['jitter_intensity_range']),
            'jpeg_quality': random.randint(*config['jpeg_quality_range']),
            'scale_factor': random.choice([2, 4]),
            'num_operations': random.randint(*config['num_operations_range'])
        }
        
        # Determine which operations to apply - NO NOISE operations
        operations = []
        prob = config['operation_probability']
        
        if random.random() < prob['iso_blur']:
            operations.append('iso_blur')
        if random.random() < prob['aniso_blur']:
            operations.append('aniso_blur')
        if random.random() < prob['jpeg_compression']:
            operations.append('jpeg_compression')
        if random.random() < prob['spatial_jitter']:
            operations.append('spatial_jitter')
        
        # Ensure at least 1 operation
        if not operations:
            operations = [random.choice(['iso_blur', 'aniso_blur'])]
        
        # Limit to max number of operations
        if len(operations) > params['num_operations']:
            operations = random.sample(operations, params['num_operations'])
        
        params['selected_operations'] = operations
        
        print(f"Auto-parameters: blur={params['blur_intensity']:.2f}, "
              f"jpeg={params['jpeg_quality']}, jitter={params['jitter_intensity']:.2f}, "
              f"ops={operations}")
        
        return params
    
    def kernel_iso(self, size, sigmaX, sigmaY):
        gauss_ax = torch.arange(size, dtype=torch.float32) - size//2
        gaussX = torch.exp(-0.5 * (gauss_ax / sigmaX) ** 2)
        gaussY = torch.exp(-0.5 * (gauss_ax / sigmaY) ** 2)
        
        gaussX /= gaussX.sum()
        gaussY /= gaussY.sum()
        
        Kernel2d = torch.outer(gaussY, gaussX)
        return Kernel2d / Kernel2d.sum()

    def kernel_aniso(self, size, sigmaX, sigmaY, rot):
        gaus_ax = torch.arange(size, dtype=torch.float32) - size//2
        
        xx, yy = torch.meshgrid(gaus_ax, gaus_ax, indexing='xy')
        
        x_rot = xx * torch.cos(rot) + yy * torch.sin(rot)
        y_rot = -xx * torch.sin(rot) + yy * torch.cos(rot)
        
        kernel2d = torch.exp(-0.5 * ((x_rot / sigmaX) ** 2 + (y_rot / sigmaY) ** 2))
        kernel2d /= kernel2d.sum()

        return kernel2d

    def B_iso_auto(self, image: torch.Tensor, params: dict) -> torch.Tensor:
        """Isotropic blur with automatic parameters"""
        c, h, w = image.shape
        kernel_size = random.choice(self.kernel_sizes)
        
        # Auto-scaled sigma based on parameters
        base_sigma_range = self.get_sigma_ranges(params['scale_factor'])[0]
        intensity = params['blur_intensity']
        sigma = random.uniform(
            base_sigma_range[0] * intensity, 
            base_sigma_range[1] * intensity
        )
        
        iso_kernel = self.kernel_iso(kernel_size, sigma, sigma)
        padding = F.pad(image.unsqueeze(0), (kernel_size // 2,) * 4, mode='reflect')
        iso_kernel = iso_kernel.expand(c, 1, kernel_size, kernel_size).to(image.device)
        blurred = F.conv2d(padding, iso_kernel, groups=c)
        
        print(f"Applied auto ISO blur: sigma={sigma:.3f}")
        return blurred.squeeze(0)

    def B_aniso_auto(self, image: torch.Tensor, params: dict) -> torch.Tensor:
        """Anisotropic blur with automatic parameters"""
        c, h, w = image.shape
        kernel_size = random.choice(self.kernel_sizes)
        kernel_size = min(kernel_size, h, w)
        if kernel_size % 2 == 0:
            kernel_size -= 1

        base_sigma_range = self.get_sigma_ranges(params['scale_factor'])[1]
        intensity = params['blur_intensity']
        
        sigma_x = random.uniform(
            base_sigma_range[0] * intensity, 
            base_sigma_range[1] * intensity
        )
        sigma_y = random.uniform(
            base_sigma_range[0] * intensity, 
            base_sigma_range[1] * intensity
        )
        rot = random.uniform(0, math.pi)

        kernel = self.kernel_aniso(kernel_size, sigma_x, sigma_y, torch.tensor(rot))
        kernel = kernel.to(dtype=torch.float32, device=image.device)
        kernel = kernel.expand(c, 1, kernel_size, kernel_size)

        image = image.to(dtype=torch.float32, device=image.device).unsqueeze(0)
        padding = kernel_size // 2
        image = F.pad(image, (padding, padding, padding, padding), mode='reflect')
        out = F.conv2d(image, kernel, groups=c)
        
        print(f"Applied auto ANISO blur: sigma_x={sigma_x:.3f}, sigma_y={sigma_y:.3f}")
        return out.squeeze(0)

    def N_jpeg_auto(self, image: torch.Tensor, params: dict) -> torch.Tensor:
        """Automatic JPEG compression"""
        quality = params['jpeg_quality']
        
        pil_image = to_pil_image(image.clamp(0, 1))
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        jpeg_img = Image.open(buffer)
        jpeg_tensor = to_tensor(jpeg_img)
        
        print(f"Applied auto JPEG: quality={quality}")
        return jpeg_tensor

    def add_jitter_auto(self, image: torch.Tensor, params: dict) -> torch.Tensor:
        """Automatic spatial jitter"""
        c, h, w = image.shape
        intensity = params['jitter_intensity']
        
        max_shift = self.jitter_shift * intensity
        shift_x = random.uniform(-max_shift, max_shift)
        shift_y = random.uniform(-max_shift, max_shift)
        
        theta = torch.tensor([
            [1, 0, shift_x],
            [0, 1, shift_y]
        ], dtype=torch.float32, device=image.device).unsqueeze(0)
        
        grid = F.affine_grid(theta, (1, c, h, w), align_corners=False)
        jittered = F.grid_sample(image.unsqueeze(0), grid, mode='bilinear', 
                                padding_mode='reflection', align_corners=False)
        
        print(f"Applied auto jitter: x={shift_x:.3f}, y={shift_y:.3f}")
        return jittered.squeeze(0)

    def degrade_auto(self, img: torch.Tensor) -> torch.Tensor:
        """Automatic degradation with NO NOISE - only blur, JPEG, and jitter"""
        # Get automatically generated parameters
        params = self.get_random_parameters()
        
        # Define operation mapping - NO NOISE operations
        operation_map = {
            'iso_blur': lambda x: self.B_iso_auto(x, params),
            'aniso_blur': lambda x: self.B_aniso_auto(x, params),
            'jpeg_compression': lambda x: self.N_jpeg_auto(x, params),
            'spatial_jitter': lambda x: self.add_jitter_auto(x, params)
        }
        
        # Get selected operations and shuffle them
        selected_ops = [operation_map[op] for op in params['selected_operations']]
        random.shuffle(selected_ops)
        
        print(f"Applying {len(selected_ops)} operations (NO NOISE): {params['selected_operations']}")
        
        # Apply operations
        for i, op in enumerate(selected_ops):
            try:
                img = op(img)
                if img is None:
                    raise ValueError(f"Operation {i+1} returned None")
            except Exception as e:
                print(f"Operation {i+1} failed: {e}")
                continue
        
        return img
    
    def save_degraded_images(self, max_workers=os.cpu_count()):
        """Save degraded PAN images as PNG files with automatic distribution"""
        os.makedirs(self.corruption_path, exist_ok=True)
        
        def process_and_save(fileinfo):
            try:
                filepath = fileinfo['path']
                filename = fileinfo['name']
                print(f"Processing PAN image: {filename}")
                
                try:
                    # Load PAN image directly
                    image = Image.open(filepath)
                    # Ensure grayscale mode for PAN images
                    if image.mode != 'L':
                        image = image.convert('L')
                except Exception as e:
                    print(f"Could not open PAN image {filepath}: {e}")
                    return f"Error opening {filename}"
                
                image_t = to_tensor(image).to(self.device)
                print(f"Original PAN shape: {image_t.shape}")
                
                # Apply automatic degradation (NO NOISE)
                degraded = self.degrade_auto(image_t)  # Changed from degrade()
                degraded = degraded.to(dtype=torch.float32, device=self.device)
                print(f"Degraded PAN shape: {degraded.shape}")
                
                img_p = to_pil_image(degraded.cpu().clamp(0, 1))

                # Convert TIFF to PNG if needed
                save_path = os.path.join(self.corruption_path, filename)
                
                # Save as PNG with lossless compression
                img_p.save(save_path, format='PNG', compress_level=6)
                
                return f"done"
    
            except Exception as e:
                return f"Error on PAN image {filename}: {e}" 
        
        # Use self.file_info_list (from base_path) directly
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_and_save, fi) for fi in self.file_info_list]
            for f in tqdm(as_completed(futures), total=len(futures), 
                         desc="Auto-Degradation Processing", unit="img"):
                result = f.result()
                if "Error" in result:
                    print(result)
    


    def rename_degraded_with_mos_and_csv(self, ref_dir, degraded_dir, csv_path, alpha=0.3):
        degraded_files = [f for f in os.listdir(degraded_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['filename', 'mos_score'])

            for fname in tqdm(degraded_files, desc="Renaming degraded images with MOS"):
                base_name, ext = os.path.splitext(fname)
                ref_path = os.path.join(ref_dir, base_name + '.tif')  # or adjust extension
                deg_path = os.path.join(degraded_dir, fname)

                ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                deg_img = cv2.imread(deg_path, cv2.IMREAD_GRAYSCALE)

                if ref_img is None or deg_img is None:
                    print(f"Could not find matching reference for {fname}")
                    continue

                mos = compute_mos(deg_img, ref_img, alpha)
                new_filename = f"{base_name}_{mos:.4f}{ext}"
                new_path = os.path.join(degraded_dir, new_filename)

                # Rename file
                os.rename(deg_path, new_path)
                writer.writerow([new_filename, mos])

        print(f"Renamed degraded images with MOS in filenames and saved scores in {csv_path}")
            
if __name__ == "__main__":
    degrador = ImageDegrador(downsample_factor_range=(2 , 4) , target_size=(512 , 512) , base_path='Data/HR' , corruption_path='Data/output')
    degrador.save_degraded_images()
    degrador.rename_degraded_with_mos_and_csv(ref_dir='Data/HR' , degraded_dir='Data/output' , csv_path='Data/MOS.csv')