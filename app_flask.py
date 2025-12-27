"""
JPEG Compression Simulator - Flask Web Interface
A lightweight web interface using Flask.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import cv2
from pathlib import Path
import base64
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from jpeg_simulator import JPEGCompressor, calculate_psnr, calculate_ssim

app = Flask(__name__)

# Create compressed_image folder
COMPRESSED_FOLDER = Path(__file__).parent / "compressed_image"
COMPRESSED_FOLDER.mkdir(exist_ok=True)


def save_to_compressed_folder(image_bgr: np.ndarray, quality: int, original_filename: str = "image") -> str:
    """Save compressed image to compressed_image folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(original_filename).stem if original_filename else "image"
    filename = f"{stem}_q{quality}_{timestamp}.jpg"
    output_path = COMPRESSED_FOLDER / filename
    cv2.imwrite(str(output_path), image_bgr)
    return str(output_path)


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', compressed_folder=str(COMPRESSED_FOLDER.absolute()))


@app.route('/compress', methods=['POST'])
def compress():
    """Handle image compression requests."""
    try:
        # Get image data
        file = request.files.get('image')
        quality = int(request.form.get('quality', 50))
        auto_save = request.form.get('auto_save', 'true').lower() == 'true'
        
        if not file:
            return jsonify({'error': 'No image provided'}), 400
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if original is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Compress
        compressor = JPEGCompressor(quality)
        compressed_data = compressor.compress(original)
        reconstructed = compressor.decompress(compressed_data)
        
        # Calculate metrics
        psnr = calculate_psnr(original, reconstructed)
        ssim = calculate_ssim(original, reconstructed)
        
        # Save to compressed_image folder if auto_save is enabled
        saved_path = None
        if auto_save:
            saved_path = save_to_compressed_folder(reconstructed, quality, file.filename)
        
        # Encode result as base64 for response
        _, buffer = cv2.imencode('.jpg', reconstructed)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'psnr': round(psnr, 2),
            'ssim': round(ssim, 4),
            'quality': quality,
            'saved_path': saved_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/saved_files')
def list_saved_files():
    """List all saved compressed files."""
    files = []
    for f in sorted(COMPRESSED_FOLDER.glob("*.jpg"), reverse=True):
        files.append({
            'name': f.name,
            'size': f.stat().st_size,
            'path': str(f)
        })
    return jsonify({
        'files': files, 
        'folder': str(COMPRESSED_FOLDER.absolute()),
        'count': len(files)
    })


@app.route('/compressed_image/<filename>')
def serve_compressed_image(filename):
    """Serve a compressed image file."""
    return send_from_directory(COMPRESSED_FOLDER, filename)


@app.route('/clear_saved', methods=['POST'])
def clear_saved():
    """Clear all saved compressed images."""
    try:
        count = 0
        for f in COMPRESSED_FOLDER.glob("*.jpg"):
            f.unlink()
            count += 1
        return jsonify({'success': True, 'deleted': count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("JPEG Compression Simulator - Flask Web Interface")
    print("=" * 60)
    print(f"Compressed images will be saved to: {COMPRESSED_FOLDER.absolute()}")
    print(f"Starting server at http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
