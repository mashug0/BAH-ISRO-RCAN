from model.BIECON import SGLCMAwareBIECON
from model.CreateDataset import CreateDataset
from model.glcm import compute_glcm_features 
from model.GradientSharpness import gradient_sharpness_metric
from model.TrainModel import train_model , compute_sharpness_batch
import torch
import pandas as pd
from model.CreateDataset import split_into_patches 
from tqdm import tqdm
import os
import cv2
from skimage.metrics import peak_signal_noise_ratio , structural_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_predict_mos(model , image , device):
    patch_size = 32
    patches = split_into_patches(image=image , patch_size= patch_size)
    patch_tensors = torch.tensor(patches, dtype=torch.float32).unsqueeze(1).to(device) / 255.0
    glcm = compute_glcm_features(patches=patch_tensors)
    sharpness = compute_sharpness_batch(patches=patch_tensors , sharpness_func=gradient_sharpness_metric)

    with torch.no_grad():
        model.eval()
        features = model(patch_tensors , glcm.to(device=device) , sharpness.to(device=device))
        pooled = model.pool_moments(features)
        predict_mos = model.regressor(pooled)
    return predict_mos.cpu().numpy().item()

def get_reference_image(df , distorted_name):
    parts = distorted_name.split('_')
    base_name = distorted_name.split("_")[0]

    file_name = f"{parts[0]}.tif"
    ref_row = df[(df['filename'].str.startswith(base_name)) & (df['score'] == 1.00)]
    if not ref_row.empty:
        return ref_row.iloc[0]['filename']
    return None

def compute_comparison(df, full_df,degraded_path , ref_path , model_path):
    model = SGLCMAwareBIECON().to(device=device)
    model.load_state_dict(torch.load(model_path))
    result_data = []
    for idx,row in tqdm(df.iterrows() , total=len(df)):
        distorted_name = row['filename']
        distorted_path = os.path.join(degraded_path , f"{distorted_name}.tif")
        refrence_name = re.sub(r'_(\d+\.\d+)$' , '' , distorted_name)
        if refrence_name is None:
            continue
        refrence_path = os.path.join(ref_path , f"{refrence_name}.tif")
        distorted = cv2.imread(distorted_path , cv2.IMREAD_GRAYSCALE)
        refrence = cv2.imread(refrence_path , cv2.IMREAD_GRAYSCALE)
        if distorted is None or refrence is None:
            continue

        psnr = peak_signal_noise_ratio(refrence , distorted)
        ssim = structural_similarity(refrence , distorted)
        predicted_mos = compute_predict_mos(model= model , image = distorted , device= device)
        result_data.append({
            'filename': distorted_name,
            'gt_mos':row['mos_score'],
            'pred_mos':predicted_mos,
            'psnr':psnr,
            'ssim':ssim
        })
    return pd.DataFrame(result_data)

def plot_comparison(df):
    plt.figure(figsize=(14, 5))

    # PSNR vs Predicted MOS
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='psnr', y='pred_mos')
    plt.title('Predicted MOS vs PSNR')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Predicted MOS')
    plt.xlim(0, 50)      # PSNR usually ranges between 10 to 50 dB
    plt.ylim(0, 1.1)     # Assuming MOS is normalized [0, 1]

    # SSIM vs Predicted MOS
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='ssim', y='pred_mos')
    plt.title('Predicted MOS vs SSIM')
    plt.xlabel('SSIM')
    plt.ylabel('Predicted MOS')
    plt.xlim(0, 1)       # SSIM always lies in [0, 1]
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.show()

    print("\nCorrelation matrix:")
    print(df[['psnr', 'ssim', 'pred_mos']].corr())


if __name__ == "__main__":
    print(device)
    print("Creating Dataset")
    
    image_dir = "C:/Users/Yashvi/OneDrive/Desktop/Dev/BAH_Blind_Eval/Data/output/"

    dataset = CreateDataset(csv_path = "C:/Users/Yashvi/OneDrive/Desktop/Dev/BAH_Blind_Eval/Data/MOS.csv",
                        image_dir = image_dir,
                        patchsize=32)
    train_len = int(len(dataset) * 0.1)
    val_len = int(len(dataset) * 0.1)
    test_len = len(dataset) - train_len - val_len

    train_dataset , val_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset,lengths= [train_len, val_len,test_len] , generator=torch.Generator().manual_seed(42))
    print("Dataset Created")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,                      # Because batch_size * num_patches affects GPU memory
        shuffle=True,                 # Parallel loading, tune per your CPU
        pin_memory=True,           # Speeds up multiple epochs
    )

    print("Loading Model")
    model = SGLCMAwareBIECON().to(device=device)
    print("Training Model")
    model.load_state_dict(torch.load('model/model_80.pth'))
    # train_model(model , train_loader ,val_dataset=val_dataset, glcm_func=compute_glcm_features , sharpness_func=gradient_sharpness_metric , epochs=30)
    # torch.save(model.state_dict() , "model/best_model.pth")
    print("Creating Test Dataset")
    test_indices = val_dataset.indices
    full_df = pd.read_csv("C:/Users/Yashvi/OneDrive/Desktop/Dev/BAH_Blind_Eval/Data/MOS.csv")
    test_df = full_df.iloc[test_indices]
    print("Computing Comparisons")
    df = compute_comparison(df = test_df , full_df=full_df , ref_path="Data/HR" ,degraded_path='Data/output', model_path="model/model_80.pth")
    print("Plotting Comparisons")
    plot_comparison(df=df)



