import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def gradient_sharpness_metric(image, sigma=1.0, edge_threshold=0.1):
    """
    Compute gradient-based edge sharpness metric.
    
    Args:
        image (np.ndarray): Input grayscale image (0-255).
        sigma (float): Gaussian blur sigma for gradient smoothing.
        edge_threshold (float): Minimum gradient magnitude to qualify as an edge.
    
    Returns:
        float: Sharpness score (higher = sharper edges).
    """
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Compute gradients
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)  # Gradient magnitude
    
    # Smooth gradients with Gaussian
    grad_mag_smooth = gaussian_filter(grad_mag, sigma=sigma)
    
    # Identify edges (thresholding)
    edge_mask = grad_mag_smooth > edge_threshold
    
    # Edge sharpness = mean gradient magnitude at edges
    sharpness = np.mean(grad_mag_smooth[edge_mask]) if np.any(edge_mask) else 0.0
    
    return sharpness
