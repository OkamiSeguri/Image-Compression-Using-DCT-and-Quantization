"""
JPEG Compression Simulator - Visualization Module
Implements visualization functions for compression analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import cv2


def plot_compression_comparison(original: np.ndarray, compressed: np.ndarray, 
                                 quality: int, save_path: Optional[str] = None):
    """
    Plot original and compressed images side by side.
    
    Args:
        original: Original image (BGR)
        compressed: Compressed image (BGR)
        quality: Quality level used
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert BGR to RGB for display
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    compressed_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
    
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(compressed_rgb)
    axes[1].set_title(f"Compressed (Q={quality})")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_dct_coefficients(dct_block: np.ndarray, title: str = "DCT Coefficients",
                          save_path: Optional[str] = None):
    """
    Visualize DCT coefficients of an 8x8 block.
    
    Args:
        dct_block: 8x8 array of DCT coefficients
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use log scale for better visualization
    display_block = np.log1p(np.abs(dct_block))
    
    im = ax.imshow(display_block, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("Horizontal Frequency")
    ax.set_ylabel("Vertical Frequency")
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label="Log magnitude")
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_quality_vs_metrics(qualities: List[int], psnr_values: List[float], 
                            ssim_values: List[float], save_path: Optional[str] = None):
    """
    Plot quality level vs PSNR and SSIM metrics.
    
    Args:
        qualities: List of quality levels
        psnr_values: List of PSNR values
        ssim_values: List of SSIM values
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # PSNR plot
    ax1.plot(qualities, psnr_values, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel("Quality Level")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Quality vs PSNR")
    ax1.grid(True, alpha=0.3)
    
    # SSIM plot
    ax2.plot(qualities, ssim_values, 'r-s', linewidth=2, markersize=8)
    ax2.set_xlabel("Quality Level")
    ax2.set_ylabel("SSIM")
    ax2.set_title("Quality vs SSIM")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_block_difference(original_block: np.ndarray, compressed_block: np.ndarray,
                          save_path: Optional[str] = None):
    """
    Visualize difference between original and compressed blocks.
    
    Args:
        original_block: Original 8x8 block
        compressed_block: Compressed 8x8 block
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(original_block, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Original Block")
    axes[0].axis('off')
    
    # Compressed
    axes[1].imshow(compressed_block, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Compressed Block")
    axes[1].axis('off')
    
    # Difference
    diff = np.abs(original_block.astype(float) - compressed_block.astype(float))
    im = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title("Absolute Difference")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()