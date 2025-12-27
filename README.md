# 🖼️ JPEG Compression Simulator

A comprehensive educational project demonstrating the **JPEG image compression pipeline** with deep focus on **Discrete Cosine Transform (DCT)** and **Quantization** algorithms. This project provides three different user interfaces: Streamlit (web), Flask (web with custom UI), and PyQt6 (desktop).

## 📋 Overview

This simulator implements the core JPEG compression steps:
1. **Input Preparation**: Supports BMP, PNG, and JPG formats with RGB, YCbCr, and Grayscale color modes
2. **Block Division**: Divides images into 8×8 pixel blocks following JPEG standard
3. **Discrete Cosine Transform (DCT)**: Converts spatial domain to frequency domain
4. **Quantization**: Lossy compression step using adjustable quality factor (1-100)
5. **Decompression**: Inverse quantization and IDCT for image reconstruction

## 🎯 Features

- **Multiple UI Options**: Choose between Streamlit, Flask, or PyQt6 interfaces
- **Adjustable Quality**: Quality factor from 1 (highest compression) to 100 (best quality)
- **Color Modes**: RGB, YCbCr (JPEG standard), or Grayscale processing
- **Visualizations**: 
  - Side-by-side original vs compressed comparison
  - DCT coefficient visualization
  - Quantized coefficient visualization
  - Quantization matrix display
- **Metrics**: 
  - Compression ratio
  - Zero coefficient percentage
  - MSE (Mean Squared Error)
  - PSNR (Peak Signal-to-Noise Ratio)
- **Export**: Save compressed images in various formats

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy opencv-python Pillow streamlit matplotlib Flask PyQt6
```

## 💻 Usage

### 1. Streamlit Web Interface (Recommended for Quick Start)

```bash
streamlit run app_streamlit.py
```

**Features:**
- Interactive sliders and controls
- Real-time visualization updates
- Clean, modern interface
- Automatic image scaling
- Built-in download functionality

**Access:** Opens automatically in browser at `http://localhost:8501`

### 2. Flask Web Application

```bash
python app_flask.py
```

**Features:**
- Custom HTML/CSS/JavaScript interface
- RESTful API backend
- Beautiful gradient design
- Responsive layout
- Session-based image handling

**Access:** Open browser to `http://localhost:5000`

### 3. PyQt6 Desktop Application

```bash
python app_pyqt6.py
```

**Features:**
- Native desktop application
- Tabbed interface for different views
- Background processing (non-blocking UI)
- File dialogs for load/save
- Detailed statistics panel

**Ideal for:** Offline use, integration with desktop workflows

## 📖 How to Use

### Basic Workflow

1. **Load Image**: Upload or select an image file (BMP, PNG, JPG)
2. **Adjust Settings**:
   - Quality Factor: 1-100 (lower = more compression, higher = better quality)
   - Color Mode: RGB, YCbCr, or Grayscale
3. **Compress**: Click the compress button to process the image
4. **Review Results**:
   - Compare original vs compressed images
   - Check compression statistics
   - View DCT and quantized coefficient visualizations
   - Examine quantization matrix
5. **Download**: Save the compressed image

### Quality Factor Guide

| Range | Description | Use Case |
|-------|-------------|----------|
| 1-25 | Extreme compression | Thumbnails, very low bandwidth |
| 26-50 | High compression | Web images, mobile apps |
| 51-75 | Balanced | General purpose, good quality/size |
| 76-90 | High quality | Professional photos, archival |
| 91-100 | Maximum quality | Medical imaging, minimal loss |

## 🔬 Technical Details

### Discrete Cosine Transform (DCT)

The 2D DCT is applied to each 8×8 block using the transformation matrix:

```
DCT(u,v) = α(u) × α(v) × Σ Σ f(x,y) × cos[(2x+1)uπ/16] × cos[(2y+1)vπ/16]
```

Where:
- `α(u) = √(1/8)` if u=0, else `√(2/8)`
- Energy is concentrated in low-frequency coefficients (top-left)

### Quantization

Coefficients are divided by a quantization matrix and rounded:

```
Q(u,v) = round(DCT(u,v) / QMatrix(u,v))
```

The quantization matrix is scaled based on quality factor:
- QF < 50: `scale = 5000 / QF`
- QF ≥ 50: `scale = 200 - 2 × QF`

### Metrics

**Compression Ratio:**
```
CR = Total Coefficients / Non-Zero Coefficients
```

**PSNR (Peak Signal-to-Noise Ratio):**
```
PSNR = 10 × log₁₀(255² / MSE)
```

Higher PSNR indicates better quality (typically 30-50 dB for good quality).

## 📁 Project Structure

```
CNN/
├── jpeg_simulator/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       ├── dct.py              # DCT and IDCT implementations
│       ├── quantization.py      # Quantization logic and metrics
│       └── compression.py       # Main compression pipeline
├── templates/
│   └── index.html              # Flask web interface
├── app_streamlit.py            # Streamlit application
├── app_flask.py                # Flask application
├── app_pyqt6.py                # PyQt6 desktop application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🎓 Educational Value

This project is designed for learning:

1. **Image Processing Fundamentals**
   - Color space conversions (RGB, YCbCr)
   - Block-based processing
   - Frequency domain analysis

2. **Compression Algorithms**
   - Lossy vs lossless compression
   - Transform coding principles
   - Quantization effects

3. **JPEG Standard**
   - Real-world compression pipeline
   - Quality vs size tradeoffs
   - Industry-standard techniques

4. **Software Development**
   - Modular code architecture
   - Multiple UI frameworks
   - Image processing libraries

## 🧪 Example Use Cases

### 1. Compare Quality Settings

Process the same image with different quality factors (10, 50, 90) to observe:
- Visual quality differences
- Compression ratios
- PSNR values
- Artifact patterns

### 2. Analyze DCT Coefficients

View DCT visualizations to understand:
- Energy concentration in low frequencies
- Effect of image content on coefficient distribution
- Block boundaries in frequency domain

### 3. Study Quantization Effects

Examine quantization matrix and quantized coefficients to see:
- How high frequencies are zeroed out
- Relationship between quality factor and coefficient retention
- Impact on compression ratio

### 4. Color Mode Comparison

Test different color modes:
- **RGB**: Full 3-channel processing
- **YCbCr**: JPEG-standard color space (human vision optimized)
- **Grayscale**: Single-channel (fastest, smallest)

## 🛠️ Advanced Usage

### Programmatic Access

```python
from jpeg_simulator.core.compression import quick_compress

# Quick compression
compressed_image, stats = quick_compress(
    'input.png',
    quality_factor=75,
    color_mode='YCbCr',
    output_path='output.png'
)

print(f"Compression Ratio: {stats['compression_ratio']:.2f}x")
print(f"PSNR: {stats['psnr']:.2f} dB")
```

### Custom Simulator

```python
from jpeg_simulator.core.compression import JPEGSimulator

# Create simulator
sim = JPEGSimulator(quality_factor=60, color_mode='RGB')

# Load and process
sim.load_image('photo.jpg')
stats = sim.compress()
result = sim.decompress()

# Access internals
dct_coeffs = sim.dct_coefficients
q_matrix = sim.q_matrix
quantized = sim.quantized_coefficients

# Save result
sim.save_compressed_image('compressed.png')
```

## 📊 Performance

- **Speed**: Processes 512×512 images in ~0.5-2 seconds (CPU-dependent)
- **Memory**: Minimal overhead, scales linearly with image size
- **Accuracy**: Implements standard JPEG DCT and quantization (minus entropy coding)

## ⚠️ Limitations

1. **Entropy Coding**: Huffman/arithmetic coding not implemented (file size is estimated)
2. **Chroma Subsampling**: 4:4:4 format only (no 4:2:0 or 4:2:2)
3. **Progressive JPEG**: Only baseline sequential mode
4. **JPEG Markers**: No file format headers (PNG/BMP output)

These limitations keep the focus on DCT and quantization algorithms while maintaining educational clarity.

## 🤝 Contributing

This is an educational project. Suggestions for improvements:
- Additional visualization options
- More color space support
- Performance optimizations
- Additional compression metrics
- Batch processing capabilities

## 📚 References

- JPEG Standard (ITU-T T.81 | ISO/IEC 10918-1)
- "JPEG Still Image Data Compression Standard" by Pennebaker & Mitchell
- Digital Image Processing by Gonzalez & Woods
- NumPy and OpenCV documentation

## 📝 License

This project is created for educational purposes. Feel free to use and modify for learning.

## 👨‍💻 Author

Created as a comprehensive CNN/Image Processing educational project demonstrating JPEG compression fundamentals.

---

**Note:** This simulator focuses on the core DCT and Quantization algorithms. For production JPEG encoding/decoding, use libraries like libjpeg, PIL/Pillow, or OpenCV.
