# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Issue: `pip install` fails
**Solutions:**
```bash
# Update pip first
python -m pip install --upgrade pip

# Try installing packages individually
pip install numpy
pip install opencv-python
pip install Pillow
pip install streamlit
pip install Flask
pip install PyQt6
```

#### Issue: `ModuleNotFoundError: No module named 'cv2'`
**Solution:**
```bash
pip install opencv-python
```

#### Issue: `ModuleNotFoundError: No module named 'PIL'`
**Solution:**
```bash
pip install Pillow
```

---

### Running Applications

#### Issue: Streamlit command not found
**Solution:**
```bash
# Make sure streamlit is installed
pip install streamlit

# Try running with python -m
python -m streamlit run app_streamlit.py
```

#### Issue: Flask app shows "Address already in use"
**Solution:**
```bash
# Port 5000 is busy, use different port
# Edit app_flask.py, change last line to:
app.run(debug=True, host='0.0.0.0', port=5001)
```

#### Issue: PyQt6 window doesn't appear
**Solutions:**
1. Check if PyQt6 is installed: `pip list | grep PyQt6`
2. Try reinstalling: `pip uninstall PyQt6` then `pip install PyQt6`
3. Check display settings (especially on Linux/Mac)

---

### Import Errors

#### Issue: `ImportError: cannot import name 'JPEGSimulator'`
**Solution:**
```bash
# Make sure you're in the correct directory
cd "d:\Python AI Code\CNN"

# Verify folder structure
dir jpeg_simulator\core  # Windows
ls jpeg_simulator/core   # Linux/Mac

# Run from project root
python app_streamlit.py
```

#### Issue: `ModuleNotFoundError: No module named 'jpeg_simulator'`
**Solution:**
The scripts add the parent directory to Python path automatically. Make sure you're running from the correct location:
```bash
# Windows
cd "d:\Python AI Code\CNN"
python demo.py

# Linux/Mac
cd "/path/to/CNN"
python demo.py
```

---

### Runtime Errors

#### Issue: "Could not load image"
**Possible Causes & Solutions:**
1. **File doesn't exist**: Check the path
2. **Unsupported format**: Use BMP, PNG, or JPG
3. **Corrupted file**: Try a different image
4. **Permissions**: Check file read permissions

#### Issue: Memory error with large images
**Solutions:**
1. Resize image before compression
2. Use grayscale mode (uses less memory)
3. Close other applications
4. Process smaller images

#### Issue: Compression is slow
**Solutions:**
1. Use grayscale mode (3× faster)
2. Reduce image size
3. Close other applications
4. Use lower quality factor (less computation)

---

### UI-Specific Issues

#### Streamlit Issues

**Issue: Browser doesn't open automatically**
```bash
# Manually open browser to:
http://localhost:8501
```

**Issue: "Streamlit requires Python 3.8+"**
```bash
# Check Python version
python --version

# Upgrade Python if needed
```

**Issue: Changes not reflected**
```bash
# Clear Streamlit cache
streamlit cache clear

# Or restart with:
# Press 'R' in terminal
# Or Ctrl+C then restart
```

#### Flask Issues

**Issue: 404 errors**
**Solution:**
Make sure `templates/` folder exists with `index.html`

**Issue: Images not displaying**
**Solution:**
Check browser console for errors (F12), verify base64 encoding

**Issue: Upload fails**
**Solution:**
Check file size (max 16MB by default), verify file format

#### PyQt6 Issues

**Issue: Window is blank/black**
**Solution:**
```bash
# Update graphics drivers
# Or try software rendering
export QT_QUICK_BACKEND=software  # Linux/Mac
set QT_QUICK_BACKEND=software     # Windows
```

**Issue: Scaling issues on high-DPI displays**
**Solution:**
Add this before creating QApplication:
```python
QApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
)
```

---

### Image Quality Issues

#### Issue: Compressed image looks very blocky
**This is expected behavior!** 
- Lower quality factors create more artifacts
- Try increasing quality factor (50-75 range)
- This demonstrates JPEG compression effects

#### Issue: PSNR is low
**Normal ranges:**
- QF 1-30: PSNR 20-30 dB (visible artifacts)
- QF 31-60: PSNR 30-35 dB (good quality)
- QF 61-90: PSNR 35-45 dB (excellent quality)
- QF 91-100: PSNR 45+ dB (near-perfect)

#### Issue: Colors look different after compression
**Causes:**
1. **Using YCbCr mode**: Color space conversion can slightly shift colors
2. **High compression**: Colors are approximated
3. **Quantization**: Lossy process affects color accuracy

**Solutions:**
- Use RGB mode for better color preservation
- Increase quality factor
- This demonstrates lossy compression effects

---

### Performance Issues

#### Issue: Processing takes too long
**Optimizations:**
```python
# 1. Use grayscale mode
simulator = JPEGSimulator(color_mode='GRAYSCALE')

# 2. Downscale image first
import cv2
image = cv2.imread('large.jpg')
image = cv2.resize(image, (512, 512))

# 3. Use lower quality (less computation)
simulator.quality_factor = 25
```

#### Issue: High memory usage
**Solutions:**
1. Process smaller images
2. Delete large variables after use
3. Use grayscale mode
4. Close visualizations when done

---

### Numerical Issues

#### Issue: `RuntimeWarning: divide by zero`
**Solution:**
This shouldn't happen, but if it does:
```python
# Ensure Q-matrix has no zeros
q_matrix = np.clip(q_matrix, 1, 255)
```

#### Issue: `RuntimeWarning: invalid value encountered`
**Solution:**
Check for NaN values:
```python
if np.isnan(image).any():
    print("Image contains NaN values")
```

---

### Testing & Validation

#### How to verify installation:
```python
# Test import
python -c "from jpeg_simulator.core.compression import JPEGSimulator; print('OK')"

# Test DCT
python -c "from jpeg_simulator.core.dct import dct_2d; import numpy as np; print(dct_2d(np.random.rand(8,8)).shape)"

# Test quantization
python -c "from jpeg_simulator.core.quantization import create_quantization_matrix; print(create_quantization_matrix(50).shape)"
```

#### Run demo to test everything:
```bash
python demo.py
```
If demo completes successfully, everything is working!

---

### Platform-Specific Issues

#### Windows

**Issue: Python command not found**
```cmd
# Use py instead
py demo.py
py -m pip install -r requirements.txt
```

**Issue: Script execution disabled**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy RemoteSigned
```

#### Linux/Mac

**Issue: Permission denied**
```bash
# Make scripts executable
chmod +x run.sh
chmod +x demo.py

# Or run with python
python demo.py
```

**Issue: Display issues with PyQt6**
```bash
# Install X11 dependencies (Linux)
sudo apt-get install python3-pyqt6
sudo apt-get install libxcb-xinerama0

# Mac: Install XQuartz if needed
brew install --cask xquartz
```

---

### Getting Help

#### Check versions:
```bash
python --version          # Should be 3.8+
pip list | grep numpy     # Should show numpy version
pip list | grep opencv    # Should show opencv-python
pip list | grep Pillow    # Should show Pillow
```

#### Verify file structure:
```bash
# Should see these directories/files:
ls -la
# - jpeg_simulator/
# - app_streamlit.py
# - app_flask.py
# - app_pyqt6.py
# - demo.py
# - requirements.txt
```

#### Clean reinstall:
```bash
# Uninstall all packages
pip uninstall -y numpy opencv-python Pillow streamlit Flask PyQt6

# Reinstall fresh
pip install -r requirements.txt

# Test
python demo.py
```

---

### Still Having Issues?

1. **Check Python version**: Must be 3.8 or higher
2. **Update all packages**: `pip install --upgrade -r requirements.txt`
3. **Try demo first**: `python demo.py` (tests everything)
4. **Check file paths**: Make sure you're in the CNN directory
5. **Read error messages**: They usually indicate the problem
6. **Try one UI at a time**: Start with Streamlit (simplest)

---

### Error Messages Decoder

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| `ModuleNotFoundError` | Missing package | `pip install <package>` |
| `FileNotFoundError` | Wrong directory | `cd` to CNN folder |
| `ValueError: Could not load` | Bad image file | Try different image |
| `MemoryError` | Image too large | Resize or use smaller image |
| `ImportError` | Path issue | Run from project root |
| `AttributeError` | Version mismatch | Update packages |

---

### Debugging Tips

**Enable verbose output:**
```python
import sys
print(f"Python: {sys.version}")
print(f"Path: {sys.path}")

import numpy as np
print(f"NumPy: {np.__version__}")

import cv2
print(f"OpenCV: {cv2.__version__}")
```

**Test minimal example:**
```python
from jpeg_simulator.core.dct import dct_2d
import numpy as np

block = np.random.rand(8, 8)
result = dct_2d(block)
print(f"Success! Result shape: {result.shape}")
```

---

**If all else fails, run demo.py - it tests everything systematically!**

```bash
python demo.py
```

This will identify exactly where the problem is.
