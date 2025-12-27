"""
JPEG Compression Simulator - Metrics Module
Implements image quality metrics.
"""

import numpy as np
import cv2


def calculate_mse(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate Mean Squared Error between two images.
    
    Args:
        original: Original image
        compressed: Compressed/reconstructed image
        
    Returns:
        MSE value
    """
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    mse = np.mean((original - compressed) ** 2)
    return float(mse)


def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        original: Original image
        compressed: Compressed/reconstructed image
        
    Returns:
        PSNR value in dB
    """
    mse = calculate_mse(original, compressed)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        original: Original image
        compressed: Compressed/reconstructed image
        
    Returns:
        SSIM value (0 to 1, where 1 is identical)
    """
    # Convert to grayscale if color
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        compressed_gray = compressed
    
    original_gray = original_gray.astype(np.float64)
    compressed_gray = compressed_gray.astype(np.float64)
    
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Mean
    mu1 = cv2.GaussianBlur(original_gray, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(compressed_gray, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Variance and covariance
    sigma1_sq = cv2.GaussianBlur(original_gray ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(compressed_gray ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(original_gray * compressed_gray, (11, 11), 1.5) - mu1_mu2
    
    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    ssim = np.mean(ssim_map)
    
    return float(ssim)