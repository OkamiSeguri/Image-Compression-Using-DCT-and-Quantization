"""
Complete JPEG Compression/Decompression Pipeline
"""

import numpy as np
import cv2
from PIL import Image

from .dct import apply_dct_to_image, apply_idct_to_image
from .quantization import (
    create_quantization_matrix,
    quantize,
    dequantize,
    calculate_compression_ratio,
    calculate_mse,
    calculate_psnr
)


class JPEGSimulator:
    """
    Main class for JPEG compression simulation.
    """
    
    def __init__(self, quality_factor=50, color_mode='RGB'):
        """
        Initialize JPEG Simulator.
        
        Args:
            quality_factor (int): Compression quality (1-100)
            color_mode (str): 'RGB', 'YCbCr', or 'GRAYSCALE'
        """
        self.quality_factor = quality_factor
        self.color_mode = color_mode
        self.original_image = None
        self.compressed_image = None
        self.dct_coefficients = None
        self.quantized_coefficients = None
        self.q_matrix = None
        self.compression_stats = {}
    
    def load_image(self, image_path):
        """
        Load image from file.
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            numpy.ndarray: Loaded image
        """
        # Load image using OpenCV
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to specified color mode
        if self.color_mode == 'GRAYSCALE':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif self.color_mode == 'YCbCr':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        
        self.original_image = image
        return image
    
    def load_image_from_array(self, image_array):
        """
        Load image from numpy array.
        
        Args:
            image_array (numpy.ndarray): Image as numpy array
        """
        self.original_image = image_array
    
    def compress(self):
        """
        Perform complete compression pipeline.
        
        Returns:
            dict: Compression statistics
        """
        if self.original_image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Step 1: Create quantization matrix
        self.q_matrix = create_quantization_matrix(self.quality_factor)
        
        # Step 2: Apply DCT
        self.dct_coefficients = apply_dct_to_image(self.original_image)
        
        # Step 3: Quantize
        self.quantized_coefficients = quantize(self.dct_coefficients, self.q_matrix)
        
        # Calculate statistics
        compression_ratio, zero_percentage = calculate_compression_ratio(
            self.dct_coefficients,
            self.quantized_coefficients
        )
        
        self.compression_stats = {
            'quality_factor': self.quality_factor,
            'compression_ratio': compression_ratio,
            'zero_percentage': zero_percentage,
            'original_shape': self.original_image.shape,
            'color_mode': self.color_mode
        }
        
        return self.compression_stats
    
    def decompress(self):
        """
        Perform complete decompression pipeline.
        
        Returns:
            numpy.ndarray: Reconstructed image
        """
        if self.quantized_coefficients is None:
            raise ValueError("No compressed data. Call compress() first.")
        
        # Step 1: Dequantize
        dequantized = dequantize(self.quantized_coefficients, self.q_matrix)
        
        # Step 2: Apply IDCT
        self.compressed_image = apply_idct_to_image(dequantized)
        
        # Calculate quality metrics
        mse = calculate_mse(self.original_image, self.compressed_image)
        psnr = calculate_psnr(self.original_image, self.compressed_image)
        
        self.compression_stats['mse'] = mse
        self.compression_stats['psnr'] = psnr
        
        return self.compressed_image
    
    def get_side_by_side_comparison(self):
        """
        Get original and compressed images side by side.
        
        Returns:
            numpy.ndarray: Combined image
        """
        if self.original_image is None or self.compressed_image is None:
            raise ValueError("Need both original and compressed images")
        
        # Ensure both images are RGB for display
        orig = self._prepare_for_display(self.original_image)
        comp = self._prepare_for_display(self.compressed_image)
        
        # Combine horizontally
        combined = np.hstack([orig, comp])
        return combined
    
    def _prepare_for_display(self, image):
        """
        Prepare image for display (convert to RGB if needed).
        
        Args:
            image (numpy.ndarray): Image to prepare
        
        Returns:
            numpy.ndarray: RGB image
        """
        if len(image.shape) == 2:
            # Grayscale to RGB
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif self.color_mode == 'YCbCr':
            # YCbCr to RGB
            return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        else:
            return image
    
    def save_compressed_image(self, output_path):
        """
        Save compressed image to file.
        
        Args:
            output_path (str): Path to save image
        """
        if self.compressed_image is None:
            raise ValueError("No compressed image. Call decompress() first.")
        
        # Prepare for saving
        save_image = self._prepare_for_display(self.compressed_image)
        
        # Convert RGB to BGR for OpenCV
        save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_path, save_image)
    
    def get_dct_visualization(self, channel=0):
        """
        Get visualization of DCT coefficients (log scale for better visibility).
        
        Args:
            channel (int): Channel to visualize (0 for grayscale or first channel)
        
        Returns:
            numpy.ndarray: Visualization image
        """
        if self.dct_coefficients is None:
            raise ValueError("No DCT coefficients. Call compress() first.")
        
        # Extract channel
        if len(self.dct_coefficients.shape) == 3:
            dct_channel = self.dct_coefficients[:, :, channel]
        else:
            dct_channel = self.dct_coefficients
        
        # Apply log transform for visualization
        dct_vis = np.log(np.abs(dct_channel) + 1)
        
        # Normalize to 0-255
        dct_vis = (dct_vis - dct_vis.min()) / (dct_vis.max() - dct_vis.min() + 1e-8)
        dct_vis = (dct_vis * 255).astype(np.uint8)
        
        return dct_vis
    
    def get_quantized_visualization(self, channel=0):
        """
        Get visualization of quantized coefficients.
        
        Args:
            channel (int): Channel to visualize (0 for grayscale or first channel)
        
        Returns:
            numpy.ndarray: Visualization image
        """
        if self.quantized_coefficients is None:
            raise ValueError("No quantized coefficients. Call compress() first.")
        
        # Extract channel
        if len(self.quantized_coefficients.shape) == 3:
            quant_channel = self.quantized_coefficients[:, :, channel]
        else:
            quant_channel = self.quantized_coefficients
        
        # Apply log transform for visualization
        quant_vis = np.log(np.abs(quant_channel) + 1)
        
        # Normalize to 0-255
        quant_vis = (quant_vis - quant_vis.min()) / (quant_vis.max() - quant_vis.min() + 1e-8)
        quant_vis = (quant_vis * 255).astype(np.uint8)
        
        return quant_vis


def quick_compress(image_path, quality_factor=50, color_mode='RGB', output_path=None):
    """
    Quick compression function for simple use cases.
    
    Args:
        image_path (str): Input image path
        quality_factor (int): Compression quality (1-100)
        color_mode (str): 'RGB', 'YCbCr', or 'GRAYSCALE'
        output_path (str): Optional output path for compressed image
    
    Returns:
        tuple: (compressed_image, statistics)
    """
    simulator = JPEGSimulator(quality_factor=quality_factor, color_mode=color_mode)
    simulator.load_image(image_path)
    stats = simulator.compress()
    compressed = simulator.decompress()
    
    if output_path:
        simulator.save_compressed_image(output_path)
    
    return compressed, stats
