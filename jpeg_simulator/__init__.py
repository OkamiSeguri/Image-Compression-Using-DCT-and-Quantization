"""
JPEG Compression Simulator Package
A comprehensive JPEG compression simulation toolkit.
"""

from .core.compressor import JPEGCompressor, simulate_compression
from .core.dct import DCTProcessor
from .core.quantization import QuantizationProcessor
from .core.color_space import ColorSpaceConverter
from .core.entropy import EntropyEncoder
from .utils.metrics import calculate_psnr, calculate_ssim, calculate_mse
from .utils.visualization import (
    plot_compression_comparison,
    plot_dct_coefficients,
    plot_quality_vs_metrics
)

__version__ = "1.0.0"
__author__ = "JPEG Simulator Team"

__all__ = [
    # Core classes
    'JPEGCompressor',
    'DCTProcessor', 
    'QuantizationProcessor',
    'ColorSpaceConverter',
    'EntropyEncoder',
    
    # Convenience functions
    'simulate_compression',
    
    # Metrics
    'calculate_psnr',
    'calculate_ssim',
    'calculate_mse',
    
    # Visualization
    'plot_compression_comparison',
    'plot_dct_coefficients',
    'plot_quality_vs_metrics',
]
