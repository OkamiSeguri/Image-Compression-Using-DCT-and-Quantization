# 🚀 Quick Start Guide

## Installation (First Time Setup)

1. **Ensure Python 3.8+ is installed**
   ```bash
   python --version
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Applications

### Option 1: Streamlit (Easiest - Web Interface)
```bash
streamlit run app_streamlit.py
```
- Opens automatically in browser
- Best for beginners
- Interactive and user-friendly

### Option 2: Flask (Custom Web UI)
```bash
python app_flask.py
```
- Open browser to: http://localhost:5000
- Beautiful custom interface
- RESTful API design

### Option 3: PyQt6 (Desktop Application)
```bash
python app_pyqt6.py
```
- Native desktop app
- Works offline
- Fast and responsive

### Option 4: Run Demo Script
```bash
python demo.py
```
- Automatically generates test images
- Tests multiple quality levels
- Creates visualizations
- No GUI required

## Quick Test

1. Run the demo script first:
   ```bash
   python demo.py
   ```
   This creates sample images you can use.

2. Then try any of the GUI applications with the generated images.

## Project Files

```
CNN/
├── jpeg_simulator/        # Core compression algorithms
├── app_streamlit.py       # Streamlit web app
├── app_flask.py          # Flask web app
├── app_pyqt6.py          # PyQt6 desktop app
├── demo.py               # Demo script
├── templates/            # Flask HTML templates
├── requirements.txt      # Dependencies
└── README.md            # Full documentation
```

## Troubleshooting

**Import Error:**
```bash
pip install --upgrade numpy opencv-python Pillow
```

**Streamlit not found:**
```bash
pip install streamlit
```

**Flask not found:**
```bash
pip install Flask
```

**PyQt6 not found:**
```bash
pip install PyQt6
```

## Need Help?

See [README.md](README.md) for complete documentation.
