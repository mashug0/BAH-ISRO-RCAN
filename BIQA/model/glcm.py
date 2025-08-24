import torch
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def compute_glcm_features(patches, distances=[1], angles=[0]):
    """
    Expects input: patches of shape [B, 1, H, W] (torch.Tensor)
    Returns: Tensor of shape [B, 6] with GLCM features (contrast, dissimilarity, homogeneity, ASM, energy, correlation)
    """
    B = patches.shape[0]
    features = []

    for i in range(B):
        img = patches[i, 0].cpu().numpy()  # [H, W]
        img = (img * 255).astype(np.uint8)  # Convert to uint8 grayscale

        glcm = graycomatrix(img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        feats = [
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'dissimilarity')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'ASM')[0, 0],
            graycoprops(glcm, 'energy')[0, 0],
            graycoprops(glcm, 'correlation')[0, 0]
        ]
        features.append(feats)

    features_tensor = torch.tensor(features, dtype=torch.float32)
    return features_tensor
