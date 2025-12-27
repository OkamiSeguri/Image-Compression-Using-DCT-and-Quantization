"""
JPEG Compression Simulator - Core Compression Module
Main class for performing JPEG-like compression on images.
"""

import numpy as np
import cv2
from pathlib import Path
import os
from .dct import DCTProcessor
from .quantization import QuantizationProcessor
from .color_space import ColorSpaceConverter
from .entropy import EntropyEncoder


class JPEGCompressor:
    """
    Main JPEG compression class that orchestrates the compression pipeline.
    """
    
    def __init__(self, quality: int = 50):
        """
        Initialize the JPEG compressor.
        
        Args:
            quality: Compression quality (1-100). Higher = better quality, larger file.
        """
        self.set_quality(quality)
        self.dct_processor = DCTProcessor()
        self.quantization_processor = QuantizationProcessor()
        self.color_converter = ColorSpaceConverter()
        self.entropy_encoder = EntropyEncoder()
        
        # Create compressed_image folder
        self.output_folder = Path(__file__).parent.parent.parent / "compressed_image"
        self.output_folder.mkdir(exist_ok=True)
    
    def set_quality(self, quality: int):
        """Set compression quality (1-100)."""
        self.quality = max(1, min(100, quality))
    
    def compress(self, image: np.ndarray) -> dict:
        """
        Compress an image using JPEG-like compression.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Dictionary containing compressed data and metadata
        """
        # Store original dimensions
        original_shape = image.shape
        
        # Step 1: Convert BGR to YCrCb color space
        ycrcb = self.color_converter.bgr_to_ycrcb(image)
        
        # Step 2: Split into channels
        y_channel = ycrcb[:, :, 0]
        cr_channel = ycrcb[:, :, 1]
        cb_channel = ycrcb[:, :, 2]
        
        # Step 3: Apply chroma subsampling (4:2:0)
        cr_subsampled = self._subsample_channel(cr_channel)
        cb_subsampled = self._subsample_channel(cb_channel)
        
        # Step 4: Apply DCT and quantization to each channel
        y_compressed = self._compress_channel(y_channel, is_luminance=True)
        cr_compressed = self._compress_channel(cr_subsampled, is_luminance=False)
        cb_compressed = self._compress_channel(cb_subsampled, is_luminance=False)
        
        # Step 5: Entropy encoding (simplified)
        compressed_data = {
            'y_data': y_compressed,
            'cr_data': cr_compressed,
            'cb_data': cb_compressed,
            'original_shape': original_shape,
            'quality': self.quality,
            'y_shape': y_channel.shape,
            'cr_shape': cr_subsampled.shape,
            'cb_shape': cb_subsampled.shape
        }
        
        return compressed_data
    
    def decompress(self, compressed_data: dict) -> np.ndarray:
        """
        Decompress JPEG-like compressed data back to an image.
        
        Args:
            compressed_data: Dictionary from compress() method
            
        Returns:
            Reconstructed image as numpy array (BGR format)
        """
        # Step 1: Inverse quantization and IDCT for each channel
        y_reconstructed = self._decompress_channel(
            compressed_data['y_data'], 
            compressed_data['y_shape'],
            is_luminance=True
        )
        cr_reconstructed = self._decompress_channel(
            compressed_data['cr_data'],
            compressed_data['cr_shape'],
            is_luminance=False
        )
        cb_reconstructed = self._decompress_channel(
            compressed_data['cb_data'],
            compressed_data['cb_shape'],
            is_luminance=False
        )
        
        # Step 2: Upsample chroma channels
        original_shape = compressed_data['original_shape']
        cr_upsampled = self._upsample_channel(cr_reconstructed, (original_shape[0], original_shape[1]))
        cb_upsampled = self._upsample_channel(cb_reconstructed, (original_shape[0], original_shape[1]))
        
        # Step 3: Merge channels
        ycrcb = np.stack([y_reconstructed, cr_upsampled, cb_upsampled], axis=2)
        
        # Step 4: Convert back to BGR
        bgr = self.color_converter.ycrcb_to_bgr(ycrcb)
        
        return bgr
    
    def compress_and_save(self, image: np.ndarray, filename: str = None) -> tuple:
        """
        Compress an image and save to the compressed_image folder.
        
        Args:
            image: Input image as numpy array (BGR format)
            filename: Optional filename for the output image
            
        Returns:
            Tuple of (saved_path, reconstructed_image)
        """
        # Compress and decompress
        compressed_data = self.compress(image)
        reconstructed = self.decompress(compressed_data)
        
        # Generate filename if not provided
        if filename is None:
            existing_files = list(self.output_folder.glob("compressed_*.jpg"))
            next_num = len(existing_files) + 1
            filename = f"compressed_{next_num:04d}_q{self.quality}.jpg"
        
        output_path = self.output_folder / filename
        
        # Save the reconstructed image
        cv2.imwrite(str(output_path), reconstructed)
        
        return str(output_path), reconstructed
    
    def compress_file(self, input_path: str, output_filename: str = None) -> tuple:
        """
        Compress an image file and save to compressed_image folder.
        
        Args:
            input_path: Path to input image
            output_filename: Optional filename for output
            
        Returns:
            Tuple of (output_path, compression_stats)
        """
        # Read input image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image: {input_path}")
        
        # Generate output filename based on input
        if output_filename is None:
            input_name = Path(input_path).stem
            output_filename = f"{input_name}_compressed_q{self.quality}.jpg"
        
        # Compress and save
        saved_path, _ = self.compress_and_save(image, output_filename)
        
        # Calculate compression statistics
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(saved_path)
        
        stats = {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': original_size / compressed_size if compressed_size > 0 else 0,
            'space_savings': (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        }
        
        return saved_path, stats
    
    def _subsample_channel(self, channel: np.ndarray) -> np.ndarray:
        """Apply 4:2:0 chroma subsampling (reduce resolution by half)."""
        return cv2.resize(channel, (channel.shape[1] // 2, channel.shape[0] // 2), 
                         interpolation=cv2.INTER_AREA)
    
    def _upsample_channel(self, channel: np.ndarray, target_size: tuple) -> np.ndarray:
        """Upsample channel back to original size."""
        return cv2.resize(channel, (target_size[1], target_size[0]), 
                         interpolation=cv2.INTER_LINEAR)
    
    def _compress_channel(self, channel: np.ndarray, is_luminance: bool = True) -> dict:
        """
        Compress a single channel using DCT and quantization.
        """
        # Pad image to be divisible by 8
        h, w = channel.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            channel = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='edge')
        
        padded_h, padded_w = channel.shape
        
        # Process 8x8 blocks
        compressed_blocks = []
        for i in range(0, padded_h, 8):
            row_blocks = []
            for j in range(0, padded_w, 8):
                block = channel[i:i+8, j:j+8].astype(np.float32)
                
                # Level shift (subtract 128)
                block = block - 128
                
                # Apply DCT
                dct_block = self.dct_processor.forward_dct(block)
                
                # Quantize
                quantized = self.quantization_processor.quantize(
                    dct_block, self.quality, is_luminance
                )
                
                row_blocks.append(quantized)
            compressed_blocks.append(row_blocks)
        
        return {
            'blocks': compressed_blocks,
            'padded_shape': (padded_h, padded_w),
            'original_shape': (h, w)
        }
    
    def _decompress_channel(self, compressed_data: dict, original_shape: tuple, 
                           is_luminance: bool = True) -> np.ndarray:
        """
        Decompress a single channel.
        """
        blocks = compressed_data['blocks']
        padded_h, padded_w = compressed_data['padded_shape']
        orig_h, orig_w = compressed_data['original_shape']
        
        # Reconstruct image from blocks
        reconstructed = np.zeros((padded_h, padded_w), dtype=np.float32)
        
        for i, row_blocks in enumerate(blocks):
            for j, quantized in enumerate(row_blocks):
                # Dequantize
                dct_block = self.quantization_processor.dequantize(
                    quantized, self.quality, is_luminance
                )
                
                # Apply inverse DCT
                block = self.dct_processor.inverse_dct(dct_block)
                
                # Level shift (add 128)
                block = block + 128
                
                # Place block in image
                reconstructed[i*8:(i+1)*8, j*8:(j+1)*8] = block
        
        # Remove padding and clip values
        reconstructed = reconstructed[:orig_h, :orig_w]
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        return reconstructed


def simulate_compression(image: np.ndarray, quality: int = 50, save: bool = False) -> np.ndarray:
    """
    Convenience function to simulate JPEG compression on an image.
    
    Args:
        image: Input image (BGR format)
        quality: Compression quality (1-100)
        save: If True, save to compressed_image folder
        
    Returns:
        Compressed and decompressed image
    """
    compressor = JPEGCompressor(quality)
    if save:
        _, reconstructed = compressor.compress_and_save(image)
        return reconstructed
    else:
        compressed = compressor.compress(image)
        return compressor.decompress(compressed)