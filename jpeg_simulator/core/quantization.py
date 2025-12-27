"""
JPEG Compression Simulator - Quantization Module
Implements quantization for JPEG compression.
"""

import numpy as np


class QuantizationProcessor:
    """
    Quantization processor for JPEG compression.
    Handles quantization and dequantization of DCT coefficients.
    """
    
    # Standard JPEG luminance quantization table
    LUMINANCE_TABLE = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)
    
    # Standard JPEG chrominance quantization table
    CHROMINANCE_TABLE = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ], dtype=np.float32)
    
    def __init__(self):
        """Initialize the quantization processor."""
        pass
    
    def get_quantization_table(self, quality: int, is_luminance: bool = True) -> np.ndarray:
        """
        Get scaled quantization table based on quality factor.
        
        Args:
            quality: Quality factor (1-100)
            is_luminance: True for luminance, False for chrominance
            
        Returns:
            8x8 quantization table
        """
        base_table = self.LUMINANCE_TABLE if is_luminance else self.CHROMINANCE_TABLE
        
        # Scale factor calculation (same as libjpeg)
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality
        
        # Apply scaling
        table = np.floor((base_table * scale + 50) / 100)
        
        # Clamp values to valid range
        table = np.clip(table, 1, 255)
        
        return table
    
    def quantize(self, dct_block: np.ndarray, quality: int, is_luminance: bool = True) -> np.ndarray:
        """
        Quantize DCT coefficients.
        
        Args:
            dct_block: 8x8 array of DCT coefficients
            quality: Quality factor (1-100)
            is_luminance: True for luminance, False for chrominance
            
        Returns:
            8x8 array of quantized coefficients
        """
        q_table = self.get_quantization_table(quality, is_luminance)
        return np.round(dct_block / q_table).astype(np.int32)
    
    def dequantize(self, quantized_block: np.ndarray, quality: int, is_luminance: bool = True) -> np.ndarray:
        """
        Dequantize coefficients back to DCT domain.
        
        Args:
            quantized_block: 8x8 array of quantized coefficients
            quality: Quality factor (1-100)
            is_luminance: True for luminance, False for chrominance
            
        Returns:
            8x8 array of dequantized DCT coefficients
        """
        q_table = self.get_quantization_table(quality, is_luminance)
        return (quantized_block * q_table).astype(np.float32)
