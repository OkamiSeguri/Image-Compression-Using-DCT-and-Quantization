# JPEG Compression Simulator - Technical Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Algorithms](#core-algorithms)
3. [Module Documentation](#module-documentation)
4. [API Reference](#api-reference)
5. [Implementation Details](#implementation-details)

---

## Architecture Overview

### System Design

```
┌─────────────────────────────────────────────┐
│           User Interfaces (UI Layer)        │
├─────────────┬───────────────┬───────────────┤
│  Streamlit  │    Flask      │    PyQt6      │
│  (Web UI)   │  (Web + JS)   │  (Desktop)    │
└─────────────┴───────────────┴───────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│         Core Compression Module             │
│        (jpeg_simulator.core)                │
├─────────────────────────────────────────────┤
│  ┌────────────┐  ┌──────────────┐          │
│  │    DCT     │  │ Quantization │          │
│  │   Module   │  │    Module    │          │
│  └────────────┘  └──────────────┘          │
│                                             │
│  ┌──────────────────────────────┐          │
│  │   Compression Pipeline       │          │
│  │   (JPEGSimulator Class)      │          │
│  └──────────────────────────────┘          │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│       Image Processing Libraries            │
│    NumPy, OpenCV, Pillow                    │
└─────────────────────────────────────────────┘
```

### Component Responsibilities

1. **UI Layer**: User interaction, file handling, visualization
2. **Core Module**: Implements DCT, quantization, and compression pipeline
3. **Image Libraries**: Low-level image operations and I/O

---

## Core Algorithms

### 1. Discrete Cosine Transform (DCT)

#### Forward DCT (Spatial → Frequency Domain)

```python
def dct_2d(block):
    """
    2D DCT using separable approach:
    DCT = T × block × T^T
    
    Where T is the 8×8 DCT transformation matrix.
    """
```

**Mathematical Formula:**

```
T[i,j] = α(i) × cos((2j + 1) × i × π / 16)

where α(i) = {
    √(1/8)  if i = 0
    √(2/8)  if i > 0
}
```

**Properties:**
- Converts spatial pixel values to frequency coefficients
- Concentrates energy in low-frequency (top-left) coefficients
- Reversible transformation (no information loss)
- Computational complexity: O(N²) per block

#### Inverse DCT (Frequency → Spatial Domain)

```python
def idct_2d(dct_block):
    """
    2D IDCT using separable approach:
    IDCT = T^T × dct_block × T
    """
```

**Properties:**
- Reconstructs spatial domain from frequency coefficients
- Same transformation matrix as forward DCT (transposed)
- Perfect reconstruction if no quantization applied

### 2. Quantization

#### Forward Quantization (Lossy Step)

```python
quantized = round(dct_coefficients / q_matrix)
```

**Quality Factor Scaling:**

```python
if quality_factor < 50:
    scale = 5000 / quality_factor
else:
    scale = 200 - 2 × quality_factor

q_matrix = clip((base_matrix × scale + 50) / 100, 1, 255)
```

**Effects:**
- High frequencies → zeros (discarded)
- Low frequencies → preserved with controlled loss
- Quality factor controls aggressiveness

#### Inverse Quantization (Dequantization)

```python
dequantized = quantized × q_matrix
```

**Note:** Information lost during quantization cannot be recovered.

### 3. Block Processing Pipeline

```
Input Image
    │
    ▼
Divide into 8×8 blocks
    │
    ▼
For each block:
│   1. Shift pixel values: pixel - 128
│   2. Apply DCT
│   3. Quantize
│   4. [Storage/Transmission]
│   5. Dequantize
│   6. Apply IDCT
│   7. Shift back: pixel + 128
│   8. Clip to [0, 255]
    │
    ▼
Reconstruct Image
```

---

## Module Documentation

### jpeg_simulator.core.dct

**Functions:**

- `dct_2d(block)`: Apply 2D DCT to 8×8 block
- `idct_2d(dct_block)`: Apply 2D IDCT to 8×8 block
- `apply_dct_to_image(image)`: Process entire image
- `apply_idct_to_image(dct_coefficients)`: Reconstruct entire image
- `apply_dct_to_channel(channel)`: Process single channel
- `apply_idct_to_channel(dct_channel)`: Reconstruct single channel

**Key Features:**
- Automatic padding for images not divisible by 8
- Multi-channel support (RGB, YCbCr)
- Efficient block-wise processing

### jpeg_simulator.core.quantization

**Constants:**

- `JPEG_STANDARD_LUMINANCE_Q_MATRIX`: Standard Y channel matrix
- `JPEG_STANDARD_CHROMINANCE_Q_MATRIX`: Standard CbCr channel matrix

**Functions:**

- `create_quantization_matrix(quality_factor, use_luminance)`: Generate Q-matrix
- `quantize(dct_coefficients, q_matrix)`: Apply quantization
- `dequantize(quantized_coefficients, q_matrix)`: Reverse quantization
- `calculate_compression_ratio(original, quantized)`: Estimate compression
- `calculate_mse(original, reconstructed)`: Compute MSE
- `calculate_psnr(original, reconstructed)`: Compute PSNR

**Quality Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MSE | `mean((orig - recon)²)` | Lower = better quality |
| PSNR | `10 × log₁₀(255² / MSE)` | Higher = better (>30 dB good) |
| CR | `total / non_zero` | Higher = more compression |

### jpeg_simulator.core.compression

**Main Class: JPEGSimulator**

```python
simulator = JPEGSimulator(
    quality_factor=50,    # 1-100
    color_mode='RGB'      # 'RGB', 'YCbCr', 'GRAYSCALE'
)
```

**Methods:**

| Method | Purpose | Returns |
|--------|---------|---------|
| `load_image(path)` | Load from file | Image array |
| `load_image_from_array(array)` | Load from memory | None |
| `compress()` | Run compression | Statistics dict |
| `decompress()` | Run decompression | Reconstructed image |
| `save_compressed_image(path)` | Save to file | None |
| `get_dct_visualization()` | Visualize DCT | Visualization image |
| `get_quantized_visualization()` | Visualize quantized | Visualization image |

**Attributes:**

- `original_image`: Input image array
- `compressed_image`: Output image array
- `dct_coefficients`: DCT frequency coefficients
- `quantized_coefficients`: Quantized coefficients
- `q_matrix`: Quantization matrix (8×8)
- `compression_stats`: Statistics dictionary

---

## API Reference

### Quick Compression Function

```python
from jpeg_simulator.core.compression import quick_compress

compressed, stats = quick_compress(
    image_path='photo.jpg',
    quality_factor=75,
    color_mode='YCbCr',
    output_path='compressed.png'
)
```

### Advanced Usage Example

```python
from jpeg_simulator.core.compression import JPEGSimulator
import numpy as np

# Create simulator
sim = JPEGSimulator(quality_factor=60, color_mode='RGB')

# Load image
sim.load_image('input.jpg')

# Compress
stats = sim.compress()
print(f"Compression: {stats['compression_ratio']:.2f}x")
print(f"Zeros: {stats['zero_percentage']:.1f}%")

# Access intermediate data
dct = sim.dct_coefficients        # Frequency domain
quantized = sim.quantized_coefficients  # After quantization
q_matrix = sim.q_matrix           # Quantization matrix

# Decompress
result = sim.decompress()
print(f"PSNR: {stats['psnr']:.2f} dB")

# Save
sim.save_compressed_image('output.png')
```

### Custom Quantization Matrix

```python
from jpeg_simulator.core.quantization import create_quantization_matrix
import numpy as np

# Create custom Q-matrix
q_matrix = create_quantization_matrix(
    quality_factor=80,
    use_luminance=True
)

# Or define completely custom matrix
custom_q = np.ones((8, 8)) * 10
custom_q[0, 0] = 1  # Preserve DC coefficient

# Use in simulator
sim.q_matrix = custom_q
```

---

## Implementation Details

### Color Space Handling

**RGB Mode:**
- Process all 3 channels independently
- Each channel: 8×8 DCT + quantization
- Best for images with color detail

**YCbCr Mode:**
- Y (luminance): Full resolution
- Cb, Cr (chrominance): Full resolution (4:4:4)
- Human vision optimized
- JPEG standard color space

**Grayscale Mode:**
- Single channel processing
- Fastest processing
- Smallest memory footprint

### Edge Padding

Images not divisible by 8 are padded:

```python
pad_height = (8 - height % 8) % 8
pad_width = (8 - width % 8) % 8
padded = np.pad(image, ((0, pad_height), (0, pad_width)), mode='edge')
```

**Padding mode:** Edge replication (repeats border pixels)

### Performance Considerations

**Time Complexity:**
- DCT per block: O(64) = O(1) constant time
- Total blocks: (H/8) × (W/8)
- Overall: O(H × W) linear in image size

**Space Complexity:**
- Original image: H × W × C
- DCT coefficients: H × W × C (float32)
- Quantized: H × W × C (float32)
- Peak memory: ~3× image size

**Optimization Tips:**
1. Use grayscale for faster processing
2. Downscale large images before compression
3. Batch process multiple images
4. Consider using numba for JIT compilation

### Numerical Precision

- Input: uint8 (0-255)
- DCT: float32 (sufficient precision)
- Quantization: round to nearest integer
- Output: uint8 (clipped to [0-255])

**Pixel Shift:**
- Before DCT: subtract 128 (center around 0)
- After IDCT: add 128 (restore range)
- Purpose: Better DCT efficiency

### Quality vs Compression Tradeoff

| QF Range | Compression | Quality | Use Case |
|----------|-------------|---------|----------|
| 1-20 | 20-50x | Poor | Extreme compression |
| 21-40 | 10-20x | Fair | Low bandwidth |
| 41-60 | 5-10x | Good | Web images |
| 61-80 | 3-5x | Very Good | Photography |
| 81-100 | 1.5-3x | Excellent | Archival |

### Limitations & Assumptions

1. **No Entropy Coding:** 
   - Actual JPEG uses Huffman/arithmetic coding
   - This implementation estimates compression ratio
   - Real files would be 2-3× smaller

2. **No Chroma Subsampling:**
   - Uses 4:4:4 (full resolution)
   - Real JPEG often uses 4:2:0 (smaller)

3. **Sequential Mode Only:**
   - No progressive JPEG support
   - All blocks processed in order

4. **Fixed Block Size:**
   - Always 8×8 (JPEG standard)
   - No adaptive block sizes

---

## Testing & Validation

### Unit Tests (Recommended)

```python
def test_dct_idct_reversibility():
    """Test that IDCT(DCT(x)) ≈ x"""
    block = np.random.rand(8, 8) * 255
    dct = dct_2d(block)
    reconstructed = idct_2d(dct)
    assert np.allclose(block, reconstructed, atol=1e-10)

def test_quantization_compression():
    """Test that quantization increases zeros"""
    dct = np.random.randn(64, 64)
    q_matrix = create_quantization_matrix(50)
    quantized = quantize(dct, q_matrix)
    zero_pct = (quantized == 0).sum() / quantized.size
    assert zero_pct > 0.3  # Should have significant zeros
```

### Validation Metrics

**Good Compression:**
- QF=50: CR ≈ 5-10x, PSNR ≈ 30-35 dB
- QF=75: CR ≈ 3-5x, PSNR ≈ 35-40 dB
- QF=90: CR ≈ 2-3x, PSNR ≈ 40-45 dB

---

## References

1. **JPEG Standard:** ITU-T Recommendation T.81 (1992)
2. **DCT Theory:** Ahmed, Natarajan, Rao (1974)
3. **Image Compression:** Salomon, Data Compression (2007)
4. **NumPy/SciPy:** Harris et al. (2020)

---

*For usage examples, see README.md and demo.py*
