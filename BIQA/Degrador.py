import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os, csv
import lpips
import torch
from torchvision import transforms
def ensure_match(img, ref):
    if img is None or ref is None:
        raise ValueError("One of the input images is None")
    # Resize to match reference
    if img.shape[:2] != ref.shape[:2]:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Convert to single channel if needed
    if len(ref.shape) == 2 and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)

def apply_degradation(img, level):
    if img is None:
        raise ValueError("Input image is None")
    if level == 0:
        return img
    elif level == 1:
        img = cv2.GaussianBlur(img, (0,0), 0.8)
        noise = np.random.normal(0, 5, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return cv2.imdecode(enc, 0)
    elif level == 2:
        img = cv2.GaussianBlur(img, (0,0), 1.5)
        noise = np.random.normal(0, 10, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return cv2.imdecode(enc, 0)
    elif level == 3:
        k = np.zeros((7,7)); k[3,:] = 1/7
        img = cv2.filter2D(img, -1, k)
        noise = np.random.normal(0, 20, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        return cv2.imdecode(enc, 0)

def gmsd(hr , sr, c = 170):
    hr = hr.astype(np.float32)
    sr = sr.astype(np.float32)

    gx_ref = cv2.Sobel(hr, cv2.CV_32F, 1, 0, ksize=3)
    gy_ref = cv2.Sobel(hr, cv2.CV_32F, 0, 1, ksize=3)
    gx_dis = cv2.Sobel(sr, cv2.CV_32F, 1, 0, ksize=3)
    gy_dis = cv2.Sobel(sr, cv2.CV_32F, 0, 1, ksize=3)

    grad_mag_ref = np.sqrt(gx_ref ** 2 + gy_ref ** 2)
    grad_mag_dis = np.sqrt(gx_dis ** 2 + gy_dis ** 2)
    
    gms = (2 * grad_mag_ref * grad_mag_dis + c) / (grad_mag_ref ** 2 + grad_mag_dis ** 2 + c)

    return np.std(gms)

def get_lpips_setting(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.stack([img]*3 , axis = 0)
    img_tensor = torch.tensor(img).unsqueeze(0)
    return img_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_model = lpips.LPIPS(net='alex').to(device).eval()

def compute_lpips(sr , hr):
    torch.no_grad()
    sr = get_lpips_setting(sr).to(device=device)
    hr = get_lpips_setting(hr).to(device=device)
    lpips_val = lpips_model(sr , hr)
    return lpips_val.item()

def compute_ssim(sr, hr):
    return ssim(sr, hr, data_range=255)
    
def compute_mos(sr, hr, beta=0.4):
    sr = ensure_match(sr, hr)
    ssim_norm = compute_ssim(sr, hr)
    lpisp = compute_lpips(sr , hr)
    return float(  (1  - beta) * ssim_norm + beta*(1-lpisp))

def build_dataset(input_dir, out_dir, alpha=0.3):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    csv_path = os.path.join(out_dir, "labels.csv")
    
    with open(csv_path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "score"])
        
        index = 0
        processed_files = 0
        
        for root, _, filenames in os.walk(input_dir):
            for filename in filenames:
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    continue
                    
                filepath = os.path.join(root, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    print(f"Failed to load {filepath}, skipping.")
                    continue
                
                base_name = os.path.splitext(filename)[0]
                
                for level in range(4):
                    try:
                        degraded = apply_degradation(img, level)
                        mos = compute_mos(degraded, img, alpha)
                        mos_str = f"{mos:.4f}"
                        
                        out_name = f"{base_name}_{level}_{mos_str}.tif"
                        out_path = os.path.join(out_dir, "images", out_name)
                        
                        if not cv2.imwrite(out_path, degraded):
                            print(f"Failed to write {out_path}, skipping.")
                            continue
                            
                        writer.writerow([out_name, mos])
                        processed_files += 1
                        
                    except Exception as e:
                        print(f"Error processing {filepath} at level {level}: {str(e)}")
                        continue
                
                index += 1
                if index % 10 == 0:
                    print(f"Processed {index} source files, created {processed_files} degraded images")
    
    print(f"Saved {processed_files} images degraded patches & labels to {out_dir}")



# Example usage
if __name__ == "__main__":
    build_dataset("C:/Users/Yashvi/OneDrive/Desktop/Dev/BAH_Blind_Eval/Data/HR", "C:/Users/Yashvi/OneDrive/Desktop/Dev/BAH_Blind_Eval/Data/output")