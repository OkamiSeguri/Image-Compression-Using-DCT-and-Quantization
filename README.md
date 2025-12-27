<<<<<<< HEAD
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
=======
# Image-Compression-Using-DCT-and-Quantization
PEG-like Image Compression Using DCT and Quantization
# JPEG Compression Simulator

A comprehensive educational and practical tool for understanding, visualizing, and analyzing JPEG image compression.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How JPEG Compression Works](#how-jpeg-compression-works)
- [Quality Metrics](#quality-metrics)
- [Screenshots](#screenshots)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **JPEG Compression Simulator** demonstrates how JPEG image compression works at a fundamental level. It provides a complete implementation of the JPEG compression pipeline with interactive visualization tools.

| Feature | Description |
|---------|-------------|
| Language | Python 3.8+ |
| Category | Image Processing / Computer Vision |
| Interfaces | CLI, Streamlit Web, Flask API, PyQt6 Desktop |
| Output | Compressed images saved to `compressed_image/` folder |

### Why Use This Tool?

- **Learn** how JPEG compression algorithm works step-by-step
- **Visualize** block-level changes and compression artifacts
- **Compare** quality settings with objective metrics
- **Optimize** images for web, storage, or printing

---

## Features

### Core Features
- Full JPEG compression/decompression pipeline
- Adjustable quality levels (1-100)
- Chroma subsampling (4:2:0)
- Standard JPEG quantization tables
- Batch image processing

### Analysis Tools
- PSNR (Peak Signal-to-Noise Ratio) calculation
- SSIM (Structural Similarity Index) calculation
- MSE (Mean Squared Error) calculation
- Block-level change detection
- Compression ratio statistics

### Visualization Features
- Side-by-side image comparison
- 8x8 block grid overlay
- Color-coded change highlighting
  - Red: High change
  - Orange: Medium-high change
  - Yellow: Medium change
- Difference heatmap (blue to red)
- Zoomed block detail view
- Quality vs metrics graphs

### Multiple Interfaces
| Interface | Command | Best For |
|-----------|---------|----------|
| Demo Script | `python demo.py` | Quick testing |
| Streamlit Web | `streamlit run app_streamlit.py` | Interactive learning |
| Flask API | `python app_flask.py` | API integration |
| PyQt6 Desktop | `python app_pyqt6.py` | Offline use |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download

```bash
cd "e:\Python code\CNN"
```

### Step 2: Install Dependencies
>>>>>>> a3f968b098a42ef18e74b42a09efe87c4457f2a3

```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
Or install manually:

```bash
pip install numpy opencv-python Pillow streamlit matplotlib Flask PyQt6
```

## 💻 Usage

### 1. Streamlit Web Interface (Recommended for Quick Start)
=======
### Step 3: Verify Installation

```bash
python -c "from jpeg_simulator import JPEGCompressor; print('Installation successful!')"
```

### Requirements File

```text
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.0.0
matplotlib>=3.4.0
streamlit>=1.0.0
flask>=2.0.0
PyQt6>=6.0.0
```

---

## Quick Start

### Option 1: Run Demo Script

```bash
python demo.py
```

This will:
- Create a sample test image
- Compress at multiple quality levels
- Display compression statistics
- Save results to `compressed_image/` folder

### Option 2: Launch Web Interface (Recommended)
>>>>>>> a3f968b098a42ef18e74b42a09efe87c4457f2a3

```bash
streamlit run app_streamlit.py
```

<<<<<<< HEAD
**Features:**
- Interactive sliders and controls
- Real-time visualization updates
- Clean, modern interface
- Automatic image scaling
- Built-in download functionality

**Access:** Opens automatically in browser at `http://localhost:8501`

### 2. Flask Web Application
=======
Then open your browser to: **http://localhost:8501**

### Option 3: Use Python API

```python
from jpeg_simulator import JPEGCompressor, calculate_psnr
import cv2

# Load image
image = cv2.imread("your_image.jpg")

# Create compressor with quality 50
compressor = JPEGCompressor(quality=50)

# Compress and decompress
compressed_data = compressor.compress(image)
result = compressor.decompress(compressed_data)

# Calculate quality metric
psnr = calculate_psnr(image, result)
print(f"PSNR: {psnr:.2f} dB")

# Save to file
compressor.compress_and_save(image, "output.jpg")
```

---

## Usage

### Command Line Demo

```bash
# Basic demo
python demo.py

# Expected output:
# ============================================================
# JPEG Compression Simulator - Demo
# ============================================================
# Creating sample test image...
# Quality  10: PSNR = 25.34 dB, SSIM = 0.8234
# Quality  30: PSNR = 29.87 dB, SSIM = 0.9012
# Quality  50: PSNR = 32.45 dB, SSIM = 0.9345
# Quality  70: PSNR = 35.67 dB, SSIM = 0.9612
# Quality  90: PSNR = 40.23 dB, SSIM = 0.9878
# ============================================================
# Demo completed!
# ============================================================
```

### Streamlit Web Interface

```bash
streamlit run app_streamlit.py
```

**Features available in web interface:**
1. Upload any image (PNG, JPG, BMP, WebP, TIFF)
2. Adjust quality slider (1-100)
3. View original vs compressed side-by-side
4. See block-level change highlighting
5. View difference heatmap
6. Examine most-changed blocks in detail
7. Download compressed images

### Flask API
>>>>>>> a3f968b098a42ef18e74b42a09efe87c4457f2a3

```bash
python app_flask.py
```

<<<<<<< HEAD
**Features:**
- Custom HTML/CSS/JavaScript interface
- RESTful API backend
- Beautiful gradient design
- Responsive layout
- Session-based image handling

**Access:** Open browser to `http://localhost:5000`

### 3. PyQt6 Desktop Application
=======
**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web page |
| `/compress` | POST | Compress an image |
| `/saved_files` | GET | List saved files |
| `/compressed_image/<filename>` | GET | Serve compressed image |

**Example API Usage:**

```python
import requests

# Compress an image via API
files = {'image': open('photo.jpg', 'rb')}
data = {'quality': 50}
response = requests.post('http://localhost:5000/compress', files=files, data=data)
result = response.json()
print(f"PSNR: {result['psnr']} dB")
```

### PyQt6 Desktop Application
>>>>>>> a3f968b098a42ef18e74b42a09efe87c4457f2a3

```bash
python app_pyqt6.py
```

**Features:**
<<<<<<< HEAD
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
=======
- Native desktop window
- Load images via file dialog
- Real-time quality adjustment
- Auto-save option
- Open output folder button

### Batch Processing

```python
from jpeg_simulator.utils.compression import batch_compress

# Compress all images in a folder
results = batch_compress(
    input_folder="input_images/",
    output_folder="output_images/",
    quality=75
)

for result in results:
    if result['success']:
        print(f"Compressed: {result['input']} -> {result['output']}")
        print(f"  Ratio: {result['stats']['compression_ratio']:.2f}x")
```

### Compare Quality Levels

```python
from jpeg_simulator.utils.compression import compare_quality_levels
import cv2

image = cv2.imread("photo.jpg")
results = compare_quality_levels(image, qualities=[10, 30, 50, 70, 90])

for r in results:
    print(f"Quality {r['quality']}: PSNR={r['psnr']:.2f} dB, SSIM={r['ssim']:.4f}")
```

---

## Project Structure

```
CNN/
├── jpeg_simulator/              # Main package
│   ├── __init__.py             # Package exports
│   ├── core/                   # Core algorithms
│   │   ├── __init__.py
│   │   ├── compressor.py       # Main JPEGCompressor class
│   │   ├── dct.py              # Discrete Cosine Transform
│   │   ├── quantization.py     # Quantization matrices
│   │   ├── color_space.py      # Color space conversion
│   │   └── entropy.py          # Entropy encoding (RLE, zigzag)
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── metrics.py          # PSNR, SSIM, MSE
│       ├── visualization.py    # Plotting functions
│       └── compression.py      # Helper functions
│
├── app_streamlit.py            # Streamlit web interface
├── app_flask.py                # Flask web API
├── app_pyqt6.py                # PyQt6 desktop app
├── demo.py                     # Demo script
│
├── templates/                  # Flask HTML templates
│   └── index.html
│
├── compressed_image/           # Output folder (auto-created)
├── sample_images/              # Sample images (auto-created)
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── PROJECT_SUMMARY.md          # Detailed documentation
├── file.md                     # Additional notes
│
├── run.bat                     # Windows launcher
└── run.sh                      # Linux/Mac launcher
```

---

## How JPEG Compression Works

### Compression Pipeline

```
Original Image
      |
      v
[1. Color Space Conversion]     BGR -> YCrCb
      |
      v
[2. Chroma Subsampling]         4:2:0 (reduce color resolution by 50%)
      |
      v
[3. Block Division]             Split into 8x8 pixel blocks
      |
      v
[4. DCT Transform]              Spatial -> Frequency domain
      |
      v
[5. Quantization]               Divide by quantization matrix (LOSSY STEP)
      |
      v
[6. Entropy Encoding]           Zigzag scan + Run-Length Encoding
      |
      v
Compressed Data
```

### Decompression Pipeline

```
Compressed Data
      |
      v
[1. Entropy Decoding]           Decode RLE data
      |
      v
[2. Dequantization]             Multiply by quantization matrix
      |
      v
[3. Inverse DCT]                Frequency -> Spatial domain
      |
      v
[4. Block Reassembly]           Combine 8x8 blocks
      |
      v
[5. Chroma Upsampling]          Restore color resolution
      |
      v
[6. Color Space Conversion]     YCrCb -> BGR
      |
      v
Reconstructed Image
```

### Quality Settings Guide

| Quality | Compression | File Size | Visual Quality | Recommended Use |
|---------|-------------|-----------|----------------|-----------------|
| 1-20 | Very High | Very Small | Poor (blocky) | Thumbnails, previews |
| 21-40 | High | Small | Acceptable | Email attachments |
| 41-60 | Medium | Medium | Good | Web images |
| 61-80 | Low | Large | Very Good | Photography |
| 81-100 | Minimal | Very Large | Excellent | Professional, archival |

---

## Quality Metrics

### PSNR (Peak Signal-to-Noise Ratio)

Measures the ratio between maximum signal power and noise power.

| PSNR Value | Quality Assessment |
|------------|-------------------|
| > 40 dB | Excellent (nearly identical) |
| 30-40 dB | Good (minor differences) |
| 20-30 dB | Acceptable (visible artifacts) |
| < 20 dB | Poor (significant degradation) |

**Formula:**
```
PSNR = 20 * log10(MAX_PIXEL / sqrt(MSE))
```

### SSIM (Structural Similarity Index)

Measures structural similarity between images (0 to 1).

| SSIM Value | Quality Assessment |
|------------|-------------------|
| > 0.95 | Excellent |
| 0.85-0.95 | Good |
| 0.70-0.85 | Acceptable |
| < 0.70 | Poor |

### MSE (Mean Squared Error)

Average of squared differences between pixels.

```
MSE = (1/n) * sum((original - compressed)^2)
```

---

## Screenshots

### Streamlit Web Interface

```
+----------------------------------------------------------+
|  JPEG Compression Simulator                               |
+----------------------------------------------------------+
|  [Upload Image]                                           |
|                                                           |
|  Quality: [==========|==========] 50                      |
|                                                           |
|  +------------------------+  +------------------------+   |
|  |                        |  |                        |   |
|  |    Original Image      |  |   Compressed Image     |   |
|  |                        |  |   (with block grid)    |   |
|  +------------------------+  +------------------------+   |
|                                                           |
|  PSNR: 32.45 dB    SSIM: 0.9345    Quality: 50           |
|                                                           |
|  Block Statistics:                                        |
|  Total: 1024 | Unchanged: 512 | Changed: 512             |
|                                                           |
|  [Download Compressed] [Download Heatmap] [Download Grid] |
+----------------------------------------------------------+
```

### Block Analysis View

```
+------------------+------------------+------------------+
|    Original      |   Compressed     |   Difference     |
|    Block         |   Block          |   (amplified)    |
+------------------+------------------+------------------+
| Block (64,128)   | MSE: 45.2        |                  |
+------------------+------------------+------------------+
```

---

## API Reference

### JPEGCompressor Class

```python
from jpeg_simulator import JPEGCompressor

# Initialize
compressor = JPEGCompressor(quality=50)

# Methods
compressor.set_quality(75)                    # Change quality
compressed_data = compressor.compress(image)  # Compress image
result = compressor.decompress(compressed_data)  # Decompress
path, stats = compressor.compress_file("input.jpg")  # File I/O
path, result = compressor.compress_and_save(image, "output.jpg")
```

### Metrics Functions

```python
from jpeg_simulator import calculate_psnr, calculate_ssim, calculate_mse

psnr = calculate_psnr(original, compressed)  # Returns dB value
ssim = calculate_ssim(original, compressed)  # Returns 0-1 value
mse = calculate_mse(original, compressed)    # Returns MSE value
```

### Utility Functions

```python
from jpeg_simulator.utils.compression import (
    compress_image,
    compress_file,
    batch_compress,
    compare_quality_levels
)

# Quick compress
result = compress_image(image, quality=50)

# Batch process
results = batch_compress("input/", "output/", quality=75)

# Compare qualities
comparison = compare_quality_levels(image, [10, 50, 90])
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: jpeg_simulator` | Package not in path | Run from CNN folder |
| `ImportError: cv2` | OpenCV not installed | `pip install opencv-python` |
| `Port 8501 in use` | Another Streamlit running | Use `--server.port 8502` |
| `Image not loading` | Unsupported format | Convert to JPG/PNG |
| `Black output image` | Image path wrong | Check file path exists |

### Debug Commands

```bash
# Check Python version
python --version

# Verify imports
python -c "from jpeg_simulator import JPEGCompressor; print('OK')"

# Check OpenCV
python -c "import cv2; print(cv2.__version__)"

# List installed packages
pip list | grep -E "numpy|opencv|streamlit|Pillow"
```

### Reinstall Dependencies

```bash
pip uninstall -y numpy opencv-python Pillow streamlit
pip install -r requirements.txt
```

---

## Contributing

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints
- Write unit tests for new features

### Reporting Issues

Please include:
- Python version
- Operating system
- Error message (full traceback)
- Steps to reproduce

---

## License

This project is open source and free to use for educational and personal purposes.

---

## Acknowledgments

- JPEG Standard: ITU-T T.81 / ISO/IEC 10918-1
- DCT Algorithm: Ahmed, Natarajan, and Rao (1974)
- SSIM Metric: Wang, Bovik, Sheikh, and Simoncelli (2004)

---

## Contact

For questions or support, please open an issue on the project repository.

---

*Version: 1.0.0*
*Last Updated: December 2025*
>>>>>>> a3f968b098a42ef18e74b42a09efe87c4457f2a3
