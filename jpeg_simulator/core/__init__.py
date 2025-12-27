"""
JPEG Compression Simulator - Core Module
Contains the main compression components.
"""

from .compressor import JPEGCompressor, simulate_compression
from .dct import DCTProcessor
from .quantization import QuantizationProcessor
from .color_space import ColorSpaceConverter
from .entropy import EntropyEncoder

__all__ = [
    'JPEGCompressor',
    'simulate_compression',
    'DCTProcessor',
    'QuantizationProcessor',
    'ColorSpaceConverter',
    'EntropyEncoder',
]
