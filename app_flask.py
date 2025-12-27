"""
JPEG Compression Simulator - Flask Web Interface
A full-featured web interface using Flask.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, session
import numpy as np
import cv2
from pathlib import Path
import base64
from datetime import datetime
import sys
import os
import uuid
import secrets

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from jpeg_simulator import JPEGCompressor, calculate_psnr, calculate_ssim

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Create folders
COMPRESSED_FOLDER = Path(__file__).parent / "compressed_image"
COMPRESSED_FOLDER.mkdir(exist_ok=True)

UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Store session data in memory
sessions = {}


def save_to_compressed_folder(image_bgr: np.ndarray, quality: int, original_filename: str = "image") -> str:
    """Save compressed image to compressed_image folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(original_filename).stem if original_filename else "image"
    filename = f"{stem}_q{quality}_{timestamp}.jpg"
    output_path = COMPRESSED_FOLDER / filename
    cv2.imwrite(str(output_path), image_bgr)
    return str(output_path)


def image_to_base64(image: np.ndarray, format: str = '.jpg') -> str:
    """Convert image to base64 string."""
    _, buffer = cv2.imencode(format, image)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"


@app.route('/')
def index():
    """Render the main page."""
    try:
        return render_template('index.html', compressed_folder=str(COMPRESSED_FOLDER.absolute()))
    except Exception as e:
        return f'''
        <!DOCTYPE html>
        <html>
        <head><title>JPEG Compression Simulator</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h1>JPEG Compression Simulator</h1>
            <p>Template file not found. Please ensure templates/index.html exists.</p>
            <p>Error: {str(e)}</p>
            <h2>API Endpoints:</h2>
            <ul>
                <li>POST /upload - Upload an image</li>
                <li>POST /compress - Compress an image</li>
                <li>POST /visualize - Generate visualizations</li>
                <li>GET /download - Download compressed image</li>
                <li>GET /saved_files - List saved files</li>
            </ul>
            <p>Output folder: {COMPRESSED_FOLDER.absolute()}</p>
        </body>
        </html>
        '''


@app.route('/upload', methods=['POST'])
def upload():
    """Handle image upload."""
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if original is None:
            return jsonify({'success': False, 'error': 'Could not decode image'}), 400
        
        # Create session
        session_id = str(uuid.uuid4())
        
        # Store in session data
        sessions[session_id] = {
            'original': original,
            'filename': file.filename,
            'compressed': None,
            'compressed_data': None
        }
        
        height, width = original.shape[:2]
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'width': width,
            'height': height,
            'filename': file.filename
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/compress', methods=['POST'])
def compress():
    """Handle image compression requests."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        quality = int(data.get('quality_factor', 50))
        color_mode = data.get('color_mode', 'rgb')
        
        if not session_id or session_id not in sessions:
            return jsonify({'success': False, 'error': 'Invalid session ID'}), 400
        
        session_data = sessions[session_id]
        original = session_data['original']
        
        # Compress
        compressor = JPEGCompressor(quality)
        compressed_data = compressor.compress(original)
        reconstructed = compressor.decompress(compressed_data)
        
        # Store results
        session_data['compressed'] = reconstructed
        session_data['compressed_data'] = compressed_data
        session_data['quality'] = quality
        
        # Calculate metrics
        psnr = calculate_psnr(original, reconstructed)
        ssim = calculate_ssim(original, reconstructed)
        
        # Get Q-matrix
        q_matrix = compressor.quantization_processor.get_quantization_table(quality).tolist()
        
        # Calculate compression stats from the actual block data
        zero_count = 0
        total_count = 0
        all_blocks = []
        
        # Collect blocks from y, cr, cb channels
        for channel_data in [compressed_data['y_data'], compressed_data['cr_data'], compressed_data['cb_data']]:
            for row_blocks in channel_data['blocks']:
                for block in row_blocks:
                    all_blocks.append(block)
                    zero_count += np.sum(block == 0)
                    total_count += block.size
        
        zero_percentage = (zero_count / total_count * 100) if total_count > 0 else 0
        
        # Calculate compression ratio estimate
        original_size = original.size
        non_zero_count = total_count - zero_count
        estimated_compressed_size = non_zero_count * 8 + 1000  # rough estimate
        compression_ratio = original_size / max(estimated_compressed_size, 1)
        
        # Save to compressed_image folder
        saved_path = save_to_compressed_folder(reconstructed, quality, session_data['filename'])
        
        # Convert images to base64
        original_b64 = image_to_base64(original)
        compressed_b64 = image_to_base64(reconstructed)
        
        return jsonify({
            'success': True,
            'original_image': original_b64,
            'compressed_image': compressed_b64,
            'stats': {
                'quality_factor': quality,
                'compression_ratio': round(compression_ratio, 2),
                'zero_percentage': round(zero_percentage, 1),
                'psnr': round(psnr, 2),
                'ssim': round(ssim, 4)
            },
            'q_matrix': q_matrix,
            'saved_path': saved_path
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/visualize', methods=['POST'])
def visualize():
    """Generate DCT and quantization visualizations."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in sessions:
            return jsonify({'success': False, 'error': 'Invalid session ID'}), 400
        
        session_data = sessions[session_id]
        compressed_data = session_data.get('compressed_data')
        
        if compressed_data is None:
            return jsonify({'success': False, 'error': 'No compression data available'}), 400
        
        # Get quantized blocks from Y channel (luminance)
        y_data = compressed_data.get('y_data', {})
        y_blocks = y_data.get('blocks', [])
        
        # Flatten the 2D list of blocks into a 1D list
        quant_blocks = []
        for row_blocks in y_blocks:
            for block in row_blocks:
                quant_blocks.append(block)
        
        # Create DCT coefficient visualization (using quantized blocks as proxy)
        if len(quant_blocks) > 0:
            # Combine blocks into an image
            num_blocks = len(quant_blocks)
            grid_size = int(np.ceil(np.sqrt(num_blocks)))
            dct_img = np.zeros((grid_size * 8, grid_size * 8))
            
            for i, block in enumerate(quant_blocks[:grid_size*grid_size]):
                row = (i // grid_size) * 8
                col = (i % grid_size) * 8
                # Normalize for visualization
                block_vis = np.log(np.abs(block) + 1)
                max_val = block_vis.max()
                block_vis = (block_vis / max_val * 255) if max_val > 0 else block_vis
                dct_img[row:row+8, col:col+8] = block_vis
            
            dct_img = dct_img.astype(np.uint8)
            dct_color = cv2.applyColorMap(dct_img, cv2.COLORMAP_JET)
            dct_b64 = image_to_base64(dct_color)
        else:
            dct_b64 = ""
        
        # Create Quantized blocks visualization
        if len(quant_blocks) > 0:
            num_blocks = len(quant_blocks)
            grid_size = int(np.ceil(np.sqrt(num_blocks)))
            quant_img = np.zeros((grid_size * 8, grid_size * 8))
            
            for i, block in enumerate(quant_blocks[:grid_size*grid_size]):
                row = (i // grid_size) * 8
                col = (i % grid_size) * 8
                # Normalize for visualization
                block_vis = np.abs(block).astype(np.float32)
                max_val = np.max(block_vis)
                if max_val > 0:
                    block_vis = (block_vis / max_val * 255)
                quant_img[row:row+8, col:col+8] = block_vis
            
            quant_img = quant_img.astype(np.uint8)
            quant_color = cv2.applyColorMap(quant_img, cv2.COLORMAP_VIRIDIS)
            quant_b64 = image_to_base64(quant_color)
        else:
            quant_b64 = ""
        
        return jsonify({
            'success': True,
            'dct_image': dct_b64,
            'quantized_image': quant_b64
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/download', methods=['GET', 'POST'])
def download():
    """Download the compressed image."""
    try:
        if request.method == 'POST':
            data = request.get_json()
            session_id = data.get('session_id') if data else None
        else:
            session_id = request.args.get('session_id')
        
        if not session_id or session_id not in sessions:
            return jsonify({'success': False, 'error': 'Invalid session ID'}), 400
        
        session_data = sessions[session_id]
        compressed = session_data.get('compressed')
        
        if compressed is None:
            return jsonify({'success': False, 'error': 'No compressed image available'}), 400
        
        # Save and return path
        quality = session_data.get('quality', 50)
        filename = session_data.get('filename', 'image')
        saved_path = save_to_compressed_folder(compressed, quality, filename)
        
        return jsonify({
            'success': True,
            'path': saved_path,
            'filename': Path(saved_path).name
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/saved_files')
def list_saved_files():
    """List all saved compressed files."""
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for f in COMPRESSED_FOLDER.glob(ext):
            files.append({
                'name': f.name,
                'size': f.stat().st_size,
                'path': str(f)
            })
    files.sort(key=lambda x: Path(x['path']).stat().st_mtime, reverse=True)
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
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for f in COMPRESSED_FOLDER.glob(ext):
                f.unlink()
                count += 1
        return jsonify({'success': True, 'deleted': count})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("JPEG Compression Simulator - Flask Web Interface")
    print("=" * 60)
    print(f"Compressed images will be saved to: {COMPRESSED_FOLDER.absolute()}")
    print(f"Starting server at http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
