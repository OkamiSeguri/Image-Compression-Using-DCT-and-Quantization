"""
JPEG Compression Simulator - Color Space Conversion Module
Implements color space conversions for JPEG compression.
"""

import numpy as np
import cv2


class ColorSpaceConverter:
    """
    Color space converter for JPEG compression.
    Handles conversions between BGR, RGB, and YCrCb color spaces.
    """
    
    def __init__(self):
        """Initialize the color space converter."""
        pass
    
    def bgr_to_ycrcb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to YCrCb color space.
        
        Args:
            image: BGR image (numpy array)
            
        Returns:
            YCrCb image (numpy array)
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    def ycrcb_to_bgr(self, image: np.ndarray) -> np.ndarray:
        """
        Convert YCrCb image to BGR color space.
        
        Args:
            image: YCrCb image (numpy array)
            
        Returns:
            BGR image (numpy array)
        """
        # Ensure proper data type
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
    
    def rgb_to_ycrcb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to YCrCb color space.
        
        Args:
            image: RGB image (numpy array)
            
        Returns:
            YCrCb image (numpy array)
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    def ycrcb_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert YCrCb image to RGB color space.
        
        Args:
            image: YCrCb image (numpy array)
            
        Returns:
            RGB image (numpy array)
        """
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
    
    def bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to RGB color space.
        
        Args:
            image: BGR image (numpy array)
            
        Returns:
            RGB image (numpy array)
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def rgb_to_bgr(self, image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to BGR color space.
        
        Args:
            image: RGB image (numpy array)
            
        Returns:
            BGR image (numpy array)
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert color image to grayscale.
        
        Args:
            image: Color image (BGR or RGB)
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)