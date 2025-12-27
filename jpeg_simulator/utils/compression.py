"""
JPEG Compression Simulator - Compression Utilities
Convenience functions for common compression tasks.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os

# Import from core module
from ..core.compressor import JPEGCompressor
from .metrics import calculate_psnr, calculate_ssim


def compress_image(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    Compress an image with specified quality.
    
    Args:
        image: Input image (BGR format)
        quality: Compression quality (1-100)
        
    Returns:
        Compressed image
    """
    compressor = JPEGCompressor(quality)
    compressed_data = compressor.compress(image)
    return compressor.decompress(compressed_data)


def compress_file(input_path: str, output_path: str = None, quality: int = 50) -> Tuple[str, Dict]:
    """
    Compress an image file.
    
    Args:
        input_path: Path to input image
        output_path: Path for output (auto-generated if None)
        quality: Compression quality
        
    Returns:
        Tuple of (output_path, stats_dict)
    """
    compressor = JPEGCompressor(quality)
    return compressor.compress_file(input_path, output_path)


def batch_compress(input_folder: str, output_folder: str = None, 
                   quality: int = 50, extensions: List[str] = None) -> List[Dict]:
    """
    Batch compress all images in a folder.
    
    Args:
        input_folder: Path to input folder
        output_folder: Path to output folder (uses compressed_image if None)
        quality: Compression quality
        extensions: List of file extensions to process
        
    Returns:
        List of compression results
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
    
    input_path = Path(input_folder)
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
    
    compressor = JPEGCompressor(quality)
    results = []
    
    for file_path in input_path.iterdir():
        if file_path.suffix.lower() in extensions:
            try:
                if output_folder:
                    output_file = output_path / f"{file_path.stem}_q{quality}.jpg"
                    saved_path, stats = compressor.compress_file(str(file_path), str(output_file))
                else:
                    saved_path, stats = compressor.compress_file(str(file_path))
                
                results.append({
                    'input': str(file_path),
                    'output': saved_path,
                    'stats': stats,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'input': str(file_path),
                    'error': str(e),
                    'success': False
                })
    
    return results


def compare_quality_levels(image: np.ndarray, 
                           qualities: List[int] = None) -> List[Dict]:
    """
    Compare compression at different quality levels.
    
    Args:
        image: Input image
        qualities: List of quality levels to test
        
    Returns:
        List of results with metrics
    """
    if qualities is None:
        qualities = [10, 20, 30, 50, 70, 80, 90, 95]
    
    compressor = JPEGCompressor()
    results = []
    
    for quality in qualities:
        compressor.set_quality(quality)
        compressed_data = compressor.compress(image)
        reconstructed = compressor.decompress(compressed_data)
        
        psnr = calculate_psnr(image, reconstructed)
        ssim = calculate_ssim(image, reconstructed)
        
        results.append({
            'quality': quality,
            'psnr': psnr,
            'ssim': ssim,
            'image': reconstructed
        })
    
    return results


def get_compression_stats(original_path: str, compressed_path: str) -> Dict:
    """
    Get compression statistics between original and compressed files.
    
    Args:
        original_path: Path to original image
        compressed_path: Path to compressed image
        
    Returns:
        Dictionary with compression statistics
    """
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    
    original = cv2.imread(original_path)
    compressed = cv2.imread(compressed_path)
    
    psnr = calculate_psnr(original, compressed)
    ssim = calculate_ssim(original, compressed)
    
    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': original_size / compressed_size if compressed_size > 0 else 0,
        'space_savings_percent': (1 - compressed_size / original_size) * 100 if original_size > 0 else 0,
        'psnr': psnr,
        'ssim': ssim
    }