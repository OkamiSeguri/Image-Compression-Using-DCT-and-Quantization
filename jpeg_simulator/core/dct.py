"""
JPEG Compression Simulator - DCT Module
Implements Discrete Cosine Transform for JPEG compression.
"""

import numpy as np


class DCTProcessor:
    """
    Discrete Cosine Transform processor for JPEG compression.
    Handles forward and inverse DCT operations on 8x8 blocks.
    """
    
    def __init__(self):
        """Initialize DCT processor with precomputed matrices."""
        self.dct_matrix = self._create_dct_matrix()
        self.dct_matrix_t = self.dct_matrix.T
    
    def _create_dct_matrix(self) -> np.ndarray:
        """
        Create the 8x8 DCT transformation matrix.
        
        Returns:
            8x8 DCT matrix
        """
        matrix = np.zeros((8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                if i == 0:
                    matrix[i, j] = 1 / np.sqrt(8)
                else:
                    matrix[i, j] = np.sqrt(2 / 8) * np.cos((2 * j + 1) * i * np.pi / 16)
        return matrix
    
    def forward_dct(self, block: np.ndarray) -> np.ndarray:
        """
        Apply forward DCT to an 8x8 block.
        
        Args:
            block: 8x8 numpy array (pixel values, typically shifted by -128)
            
        Returns:
            8x8 array of DCT coefficients
        """
        return self.dct_matrix @ block @ self.dct_matrix_t
    
    def inverse_dct(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Apply inverse DCT to reconstruct an 8x8 block.
        
        Args:
            coefficients: 8x8 array of DCT coefficients
            
        Returns:
            8x8 array of reconstructed pixel values
        """
        return self.dct_matrix_t @ coefficients @ self.dct_matrix
    
    def forward_dct_2d(self, image: np.ndarray) -> np.ndarray:
        """
        Apply forward DCT to entire image (block by block).
        
        Args:
            image: 2D numpy array (must be divisible by 8)
            
        Returns:
            2D array of DCT coefficients
        """
        h, w = image.shape
        result = np.zeros_like(image, dtype=np.float32)
        
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = image[i:i+8, j:j+8].astype(np.float32)
                result[i:i+8, j:j+8] = self.forward_dct(block)
        
        return result
    
    def inverse_dct_2d(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Apply inverse DCT to entire coefficient array (block by block).
        
        Args:
            coefficients: 2D array of DCT coefficients
            
        Returns:
            2D array of reconstructed pixel values
        """
        h, w = coefficients.shape
        result = np.zeros_like(coefficients, dtype=np.float32)
        
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = coefficients[i:i+8, j:j+8]
                result[i:i+8, j:j+8] = self.inverse_dct(block)
        
        return result
