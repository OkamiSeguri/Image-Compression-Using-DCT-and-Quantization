"""
Complete JPEG Compression/Decompression Pipeline
Alternative interface for JPEG compression simulation.
"""

import numpy as np
import cv2
from pathlib import Path

from .dct import DCTProcessor
from .quantization import QuantizationProcessor
from .color_space import ColorSpaceConverter


class JPEGSimulator:
    """
    Alternative JPEG compression simulator class.
    Provides a different interface for compression tasks.
    """
    
    def __init__(self, quality_factor=50, color_mode='RGB'):
        """
        Initialize JPEG Simulator.
        
        Args:
            quality_factor (int): Compression quality (1-100)
            color_mode (str): 'RGB', 'YCbCr', or 'GRAYSCALE'
        """
        self.quality_factor = max(1, min(100, quality_factor))
        self.color_mode = color_mode
        self.original_image = None
        self.compressed_image = None
        self.dct_coefficients = None
        self.quantized_coefficients = None
        self.compression_stats = {}
        
        # Initialize processors
        self.dct_processor = DCTProcessor()
        self.quantization_processor = QuantizationProcessor()
        self.color_converter = ColorSpaceConverter()
    
    def load_image(self, image_path):
        """
        Load image from file.
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            numpy.ndarray: Loaded image
        """
        # Load image using OpenCV
        image = cv2.imread(str(image_path))
        
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
        self.original_image = image_array.copy()
    
    def compress(self):
        """
        Perform complete compression pipeline.
        
        Returns:
            dict: Compression statistics
        """
        if self.original_image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        image = self.original_image
        
        # Handle different image types
        if len(image.shape) == 2:
            # Grayscale
            self.dct_coefficients = self._compress_channel(image)
            self.quantized_coefficients = self._quantize_channel(self.dct_coefficients)
        else:
            # Color image - process each channel
            channels_dct = []
            channels_quant = []
            for c in range(image.shape[2]):
                dct_c = self._compress_channel(image[:, :, c])
                quant_c = self._quantize_channel(dct_c)
                channels_dct.append(dct_c)
                channels_quant.append(quant_c)
            
            self.dct_coefficients = np.stack(channels_dct, axis=2)
            self.quantized_coefficients = np.stack(channels_quant, axis=2)
        
        # Calculate statistics
        total_coeffs = self.quantized_coefficients.size
        zero_coeffs = np.sum(self.quantized_coefficients == 0)
        zero_percentage = (zero_coeffs / total_coeffs) * 100
        
        self.compression_stats = {
            'quality_factor': self.quality_factor,
            'zero_percentage': zero_percentage,
            'original_shape': self.original_image.shape,
            'color_mode': self.color_mode
        }
        
        return self.compression_stats
    
    def _compress_channel(self, channel):
        """Apply DCT to a single channel."""
        h, w = channel.shape
        
        # Pad to multiple of 8
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            channel = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='edge')
        
        padded_h, padded_w = channel.shape
        result = np.zeros_like(channel, dtype=np.float32)
        
        for i in range(0, padded_h, 8):
            for j in range(0, padded_w, 8):
                block = channel[i:i+8, j:j+8].astype(np.float32) - 128
                result[i:i+8, j:j+8] = self.dct_processor.forward_dct(block)
        
        return result[:h, :w]
    
    def _quantize_channel(self, dct_channel):
        """Quantize DCT coefficients."""
        h, w = dct_channel.shape
        
        # Pad to multiple of 8
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            dct_channel = np.pad(dct_channel, ((0, pad_h), (0, pad_w)), mode='constant')
        
        padded_h, padded_w = dct_channel.shape
        result = np.zeros_like(dct_channel, dtype=np.int32)
        
        for i in range(0, padded_h, 8):
            for j in range(0, padded_w, 8):
                block = dct_channel[i:i+8, j:j+8]
                result[i:i+8, j:j+8] = self.quantization_processor.quantize(
                    block, self.quality_factor, is_luminance=True
                )
        
        return result[:h, :w]
    
    def decompress(self):
        """
        Perform complete decompression pipeline.
        
        Returns:
            numpy.ndarray: Reconstructed image
        """
        if self.quantized_coefficients is None:
            raise ValueError("No compressed data. Call compress() first.")
        
        quant = self.quantized_coefficients
        
        # Handle different image types
        if len(quant.shape) == 2:
            # Grayscale
            self.compressed_image = self._decompress_channel(quant)
        else:
            # Color image
            channels = []
            for c in range(quant.shape[2]):
                channel = self._decompress_channel(quant[:, :, c])
                channels.append(channel)
            self.compressed_image = np.stack(channels, axis=2)
        
        # Calculate quality metrics
        mse = np.mean((self.original_image.astype(np.float64) - 
                       self.compressed_image.astype(np.float64)) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255**2 / mse)
        else:
            psnr = float('inf')
        
        self.compression_stats['mse'] = float(mse)
        self.compression_stats['psnr'] = float(psnr)
        
        return self.compressed_image
    
    def _decompress_channel(self, quant_channel):
        """Decompress a single channel."""
        h, w = quant_channel.shape
        
        # Pad to multiple of 8
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            quant_channel = np.pad(quant_channel, ((0, pad_h), (0, pad_w)), mode='constant')
        
        padded_h, padded_w = quant_channel.shape
        result = np.zeros_like(quant_channel, dtype=np.float32)
        
        for i in range(0, padded_h, 8):
            for j in range(0, padded_w, 8):
                block = quant_channel[i:i+8, j:j+8]
                dequant = self.quantization_processor.dequantize(
                    block, self.quality_factor, is_luminance=True
                )
                result[i:i+8, j:j+8] = self.dct_processor.inverse_dct(dequant) + 128
        
        result = np.clip(result[:h, :w], 0, 255).astype(np.uint8)
        return result
    
    def get_side_by_side_comparison(self):
        """
        Get original and compressed images side by side.
        
        Returns:
            numpy.ndarray: Combined image
        """
        if self.original_image is None or self.compressed_image is None:
            raise ValueError("Need both original and compressed images")
        
        orig = self._prepare_for_display(self.original_image)
        comp = self._prepare_for_display(self.compressed_image)
        
        combined = np.hstack([orig, comp])
        return combined
    
    def _prepare_for_display(self, image):
        """Prepare image for display (convert to RGB if needed)."""
        if len(image.shape) == 2:
            return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif self.color_mode == 'YCbCr':
            return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        else:
            return image.astype(np.uint8)
    
    def save_compressed_image(self, output_path):
        """
        Save compressed image to file.
        
        Args:
            output_path (str): Path to save image
        """
        if self.compressed_image is None:
            raise ValueError("No compressed image. Call decompress() first.")
        
        save_image = self._prepare_for_display(self.compressed_image)
        save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), save_image)
    
    def get_dct_visualization(self, channel=0):
        """
        Get visualization of DCT coefficients.
        
        Args:
            channel (int): Channel to visualize
        
        Returns:
            numpy.ndarray: Visualization image
        """
        if self.dct_coefficients is None:
            raise ValueError("No DCT coefficients. Call compress() first.")
        
        if len(self.dct_coefficients.shape) == 3:
            dct_channel = self.dct_coefficients[:, :, channel]
        else:
            dct_channel = self.dct_coefficients
        
        dct_vis = np.log(np.abs(dct_channel) + 1)
        dct_vis = (dct_vis - dct_vis.min()) / (dct_vis.max() - dct_vis.min() + 1e-8)
        dct_vis = (dct_vis * 255).astype(np.uint8)
        
        return dct_vis
    
    def get_quantized_visualization(self, channel=0):
        """
        Get visualization of quantized coefficients.
        
        Args:
            channel (int): Channel to visualize
        
        Returns:
            numpy.ndarray: Visualization image
        """
        if self.quantized_coefficients is None:
            raise ValueError("No quantized coefficients. Call compress() first.")
        
        if len(self.quantized_coefficients.shape) == 3:
            quant_channel = self.quantized_coefficients[:, :, channel]
        else:
            quant_channel = self.quantized_coefficients
        
        quant_vis = np.log(np.abs(quant_channel) + 1)
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
