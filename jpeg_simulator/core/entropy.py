"""
JPEG Compression Simulator - Entropy Encoding Module
Implements entropy encoding techniques used in JPEG compression.
"""

import numpy as np
from typing import List, Tuple


class EntropyEncoder:
    """
    Entropy encoder for JPEG compression.
    Implements Run-Length Encoding (RLE) and prepares data for Huffman coding.
    """
    
    def __init__(self):
        """Initialize the entropy encoder."""
        self.zigzag_order = self._create_zigzag_order()
    
    def _create_zigzag_order(self) -> List[Tuple[int, int]]:
        """
        Create zigzag scan order for 8x8 block.
        
        Returns:
            List of (row, col) tuples in zigzag order
        """
        order = []
        for s in range(15):  # Sum of indices ranges from 0 to 14
            if s % 2 == 0:
                # Even sum: go down-left
                for i in range(min(s, 7), max(0, s - 7) - 1, -1):
                    j = s - i
                    if 0 <= i < 8 and 0 <= j < 8:
                        order.append((i, j))
            else:
                # Odd sum: go up-right
                for i in range(max(0, s - 7), min(s, 7) + 1):
                    j = s - i
                    if 0 <= i < 8 and 0 <= j < 8:
                        order.append((i, j))
        return order
    
    def zigzag_scan(self, block: np.ndarray) -> np.ndarray:
        """
        Convert 8x8 block to 1D array using zigzag scan.
        
        Args:
            block: 8x8 numpy array
            
        Returns:
            1D array of 64 elements in zigzag order
        """
        result = np.zeros(64, dtype=block.dtype)
        for idx, (i, j) in enumerate(self.zigzag_order):
            result[idx] = block[i, j]
        return result
    
    def inverse_zigzag(self, array: np.ndarray) -> np.ndarray:
        """
        Convert 1D zigzag array back to 8x8 block.
        
        Args:
            array: 1D array of 64 elements
            
        Returns:
            8x8 numpy array
        """
        block = np.zeros((8, 8), dtype=array.dtype)
        for idx, (i, j) in enumerate(self.zigzag_order):
            block[i, j] = array[idx]
        return block
    
    def run_length_encode(self, array: np.ndarray) -> List[Tuple[int, int]]:
        """
        Apply run-length encoding to zigzag-scanned array.
        Encodes (skip, value) pairs where skip is the number of zeros before value.
        
        Args:
            array: 1D array (typically from zigzag scan)
            
        Returns:
            List of (zero_run_length, value) tuples
        """
        encoded = []
        zero_count = 0
        
        for value in array:
            if value == 0:
                zero_count += 1
            else:
                # Handle runs longer than 15 zeros (JPEG limit)
                while zero_count > 15:
                    encoded.append((15, 0))  # ZRL (Zero Run Length) symbol
                    zero_count -= 16
                encoded.append((zero_count, int(value)))
                zero_count = 0
        
        # End of block marker
        if zero_count > 0:
            encoded.append((0, 0))  # EOB (End of Block)
        
        return encoded
    
    def run_length_decode(self, encoded: List[Tuple[int, int]], length: int = 64) -> np.ndarray:
        """
        Decode run-length encoded data back to array.
        
        Args:
            encoded: List of (zero_run_length, value) tuples
            length: Expected length of output array
            
        Returns:
            Decoded 1D array
        """
        result = np.zeros(length, dtype=np.float32)
        position = 0
        
        for zero_run, value in encoded:
            if zero_run == 0 and value == 0:
                # EOB - rest is zeros
                break
            position += zero_run
            if position < length:
                result[position] = value
                position += 1
        
        return result
    
    def encode_block(self, block: np.ndarray) -> dict:
        """
        Encode an 8x8 block using zigzag scan and RLE.
        
        Args:
            block: 8x8 quantized DCT coefficient block
            
        Returns:
            Dictionary with DC coefficient and AC RLE data
        """
        zigzag = self.zigzag_scan(block)
        
        # Separate DC and AC coefficients
        dc = int(zigzag[0])
        ac = zigzag[1:]
        
        # RLE encode AC coefficients
        ac_encoded = self.run_length_encode(ac)
        
        return {
            'dc': dc,
            'ac': ac_encoded
        }
    
    def decode_block(self, encoded: dict) -> np.ndarray:
        """
        Decode encoded block data back to 8x8 block.
        
        Args:
            encoded: Dictionary with DC and AC data
            
        Returns:
            8x8 numpy array
        """
        # Decode AC coefficients
        ac = self.run_length_decode(encoded['ac'], length=63)
        
        # Combine DC and AC
        zigzag = np.zeros(64, dtype=np.float32)
        zigzag[0] = encoded['dc']
        zigzag[1:] = ac
        
        # Convert back to 8x8 block
        return self.inverse_zigzag(zigzag)
    
    def encode_dc_differences(self, dc_values: List[int]) -> List[int]:
        """
        Encode DC coefficients as differences (DPCM).
        
        Args:
            dc_values: List of DC coefficients
            
        Returns:
            List of DC differences
        """
        if not dc_values:
            return []
        
        differences = [dc_values[0]]
        for i in range(1, len(dc_values)):
            differences.append(dc_values[i] - dc_values[i-1])
        
        return differences
    
    def decode_dc_differences(self, differences: List[int]) -> List[int]:
        """
        Decode DC differences back to actual values.
        
        Args:
            differences: List of DC differences
            
        Returns:
            List of DC values
        """
        if not differences:
            return []
        
        dc_values = [differences[0]]
        for i in range(1, len(differences)):
            dc_values.append(dc_values[i-1] + differences[i])
        
        return dc_values