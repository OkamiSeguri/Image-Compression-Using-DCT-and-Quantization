"""
JPEG Compression Simulator - Streamlit Web Interface
A user-friendly web interface for the JPEG compression simulator.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from jpeg_simulator import JPEGCompressor, calculate_psnr, calculate_ssim

# Page configuration
st.set_page_config(
    page_title="JPEG Compression Simulator",
    page_icon="image",
    layout="wide"
)

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


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy array (BGR) to PIL Image (RGB)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (BGR)."""
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def create_difference_map(original: np.ndarray, compressed: np.ndarray) -> np.ndarray:
    """
    Create a color-coded difference map between original and compressed images.
    Green = no change, Yellow = small change, Red = large change
    """
    # Calculate absolute difference
    diff = np.abs(original.astype(np.float32) - compressed.astype(np.float32))
    
    # Convert to grayscale difference
    if len(diff.shape) == 3:
        diff_gray = np.mean(diff, axis=2)
    else:
        diff_gray = diff
    
    # Normalize to 0-255
    diff_normalized = np.clip(diff_gray * 3, 0, 255).astype(np.uint8)
    
    # Apply colormap (COLORMAP_JET: blue=low, red=high)
    diff_colored = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
    
    return diff_colored


def create_block_grid_overlay(image: np.ndarray, block_size: int = 8, 
                               color: tuple = (0, 255, 0), thickness: int = 1) -> np.ndarray:
    """
    Create an image with 8x8 block grid overlay.
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    # Draw vertical lines
    for x in range(0, w, block_size):
        cv2.line(result, (x, 0), (x, h), color, thickness)
    
    # Draw horizontal lines
    for y in range(0, h, block_size):
        cv2.line(result, (0, y), (w, y), color, thickness)
    
    return result


def highlight_changed_blocks(original: np.ndarray, compressed: np.ndarray, 
                              block_size: int = 8, threshold: float = 5.0) -> np.ndarray:
    """
    Highlight blocks that have significant changes.
    Returns image with colored overlays on changed blocks.
    """
    result = compressed.copy()
    h, w = original.shape[:2]
    
    # Create overlay for highlighting
    overlay = result.copy()
    
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            # Extract blocks
            orig_block = original[y:y+block_size, x:x+block_size]
            comp_block = compressed[y:y+block_size, x:x+block_size]
            
            # Calculate block difference (MSE)
            block_diff = np.mean((orig_block.astype(np.float32) - comp_block.astype(np.float32)) ** 2)
            
            # Color based on difference level
            if block_diff > threshold * 10:
                # High difference - Red
                color = (0, 0, 255)
                alpha = 0.4
            elif block_diff > threshold * 5:
                # Medium-high difference - Orange
                color = (0, 128, 255)
                alpha = 0.3
            elif block_diff > threshold:
                # Medium difference - Yellow
                color = (0, 255, 255)
                alpha = 0.2
            else:
                # Low difference - skip
                continue
            
            # Draw filled rectangle on overlay
            cv2.rectangle(overlay, (x, y), (x + block_size, y + block_size), color, -1)
    
    # Blend overlay with result
    result = cv2.addWeighted(overlay, 0.3, result, 0.7, 0)
    
    # Draw grid lines
    result = create_block_grid_overlay(result, block_size, (128, 128, 128), 1)
    
    return result


def create_side_by_side_blocks(original: np.ndarray, compressed: np.ndarray,
                                block_coords: list, block_size: int = 8) -> np.ndarray:
    """
    Create a comparison view of specific blocks side by side.
    """
    num_blocks = len(block_coords)
    if num_blocks == 0:
        return None
    
    # Calculate output size
    scale = 8  # Scale up blocks for visibility
    block_display_size = block_size * scale
    margin = 10
    
    # Create output image
    output_width = (block_display_size * 3 + margin * 4) * min(num_blocks, 4)
    output_height = ((num_blocks + 3) // 4) * (block_display_size + margin * 2 + 30)
    output = np.ones((max(output_height, 100), max(output_width, 100), 3), dtype=np.uint8) * 240
    
    for idx, (x, y) in enumerate(block_coords[:8]):  # Limit to 8 blocks
        row = idx // 4
        col = idx % 4
        
        base_x = col * (block_display_size * 3 + margin * 4) + margin
        base_y = row * (block_display_size + margin * 2 + 30) + margin
        
        # Extract blocks
        orig_block = original[y:y+block_size, x:x+block_size]
        comp_block = compressed[y:y+block_size, x:x+block_size]
        diff_block = np.abs(orig_block.astype(np.float32) - comp_block.astype(np.float32))
        diff_block = np.clip(diff_block * 3, 0, 255).astype(np.uint8)
        
        # Scale up blocks
        orig_scaled = cv2.resize(orig_block, (block_display_size, block_display_size), 
                                  interpolation=cv2.INTER_NEAREST)
        comp_scaled = cv2.resize(comp_block, (block_display_size, block_display_size), 
                                  interpolation=cv2.INTER_NEAREST)
        diff_scaled = cv2.resize(diff_block, (block_display_size, block_display_size), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Place blocks in output
        y_pos = base_y + 20
        
        # Original block
        x_pos = base_x
        output[y_pos:y_pos+block_display_size, x_pos:x_pos+block_display_size] = orig_scaled
        
        # Compressed block
        x_pos = base_x + block_display_size + margin
        output[y_pos:y_pos+block_display_size, x_pos:x_pos+block_display_size] = comp_scaled
        
        # Difference block
        x_pos = base_x + (block_display_size + margin) * 2
        output[y_pos:y_pos+block_display_size, x_pos:x_pos+block_display_size] = diff_scaled
        
        # Add label
        label = f"Block ({x},{y})"
        cv2.putText(output, label, (base_x, base_y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return output


def find_most_changed_blocks(original: np.ndarray, compressed: np.ndarray, 
                              block_size: int = 8, num_blocks: int = 8) -> list:
    """
    Find the blocks with the most significant changes.
    """
    h, w = original.shape[:2]
    block_diffs = []
    
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            orig_block = original[y:y+block_size, x:x+block_size]
            comp_block = compressed[y:y+block_size, x:x+block_size]
            
            # Calculate MSE for this block
            mse = np.mean((orig_block.astype(np.float32) - comp_block.astype(np.float32)) ** 2)
            block_diffs.append((mse, x, y))
    
    # Sort by MSE (descending) and return top blocks
    block_diffs.sort(reverse=True)
    return [(x, y) for _, x, y in block_diffs[:num_blocks]]


def create_block_statistics(original: np.ndarray, compressed: np.ndarray, 
                             block_size: int = 8) -> dict:
    """
    Calculate statistics about block-level changes.
    """
    h, w = original.shape[:2]
    
    total_blocks = 0
    unchanged_blocks = 0
    low_change_blocks = 0
    medium_change_blocks = 0
    high_change_blocks = 0
    
    threshold_low = 5
    threshold_medium = 25
    threshold_high = 100
    
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            orig_block = original[y:y+block_size, x:x+block_size]
            comp_block = compressed[y:y+block_size, x:x+block_size]
            
            mse = np.mean((orig_block.astype(np.float32) - comp_block.astype(np.float32)) ** 2)
            total_blocks += 1
            
            if mse < threshold_low:
                unchanged_blocks += 1
            elif mse < threshold_medium:
                low_change_blocks += 1
            elif mse < threshold_high:
                medium_change_blocks += 1
            else:
                high_change_blocks += 1
    
    return {
        'total_blocks': total_blocks,
        'unchanged': unchanged_blocks,
        'low_change': low_change_blocks,
        'medium_change': medium_change_blocks,
        'high_change': high_change_blocks,
        'unchanged_percent': (unchanged_blocks / total_blocks * 100) if total_blocks > 0 else 0,
        'changed_percent': ((total_blocks - unchanged_blocks) / total_blocks * 100) if total_blocks > 0 else 0
    }


def main():
    st.title("JPEG Compression Simulator")
    st.markdown("Upload an image to simulate JPEG compression and analyze block-level changes.")
    st.info(f"All compressed images are saved to: `{COMPRESSED_FOLDER.absolute()}`")
    
    # Sidebar controls
    st.sidebar.header("Compression Settings")
    quality = st.sidebar.slider(
        "Quality Level", 
        min_value=1, 
        max_value=100, 
        value=50,
        help="Higher values = better quality, larger file size"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("Block Analysis Settings")
    
    block_size = st.sidebar.selectbox(
        "Block Size",
        options=[8, 16, 32],
        index=0,
        help="Size of blocks for analysis (JPEG uses 8x8)"
    )
    
    diff_threshold = st.sidebar.slider(
        "Change Threshold",
        min_value=1.0,
        max_value=50.0,
        value=5.0,
        help="Threshold for highlighting changed blocks"
    )
    
    show_grid = st.sidebar.checkbox("Show Block Grid", value=True)
    show_diff_map = st.sidebar.checkbox("Show Difference Heatmap", value=True)
    show_block_details = st.sidebar.checkbox("Show Block Details", value=True)
    
    auto_save = st.sidebar.checkbox("Auto-save compressed images", value=False)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff']
    )
    
    if uploaded_file is not None:
        # Read image
        pil_image = Image.open(uploaded_file).convert('RGB')
        original = pil_to_numpy(pil_image)
        
        # Create compressor and process
        compressor = JPEGCompressor(quality)
        compressed_data = compressor.compress(original)
        reconstructed = compressor.decompress(compressed_data)
        
        # Calculate metrics
        psnr = calculate_psnr(original, reconstructed)
        ssim = calculate_ssim(original, reconstructed)
        
        # Auto-save if enabled
        if auto_save:
            saved_path = save_to_compressed_folder(reconstructed, quality, uploaded_file.name)
            st.success(f"Compressed image saved to: `{saved_path}`")
        
        # Main metrics display
        st.subheader("Quality Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Quality Level", f"{quality}")
        with metric_col2:
            st.metric("PSNR", f"{psnr:.2f} dB")
        with metric_col3:
            st.metric("SSIM", f"{ssim:.4f}")
        with metric_col4:
            st.metric("Image Size", f"{original.shape[1]}x{original.shape[0]}")
        
        # Block statistics
        block_stats = create_block_statistics(original, reconstructed, block_size)
        
        st.subheader("Block-Level Statistics")
        stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
        
        with stat_col1:
            st.metric("Total Blocks", block_stats['total_blocks'])
        with stat_col2:
            st.metric("Unchanged", block_stats['unchanged'], 
                     delta=f"{block_stats['unchanged_percent']:.1f}%")
        with stat_col3:
            st.metric("Low Change", block_stats['low_change'])
        with stat_col4:
            st.metric("Medium Change", block_stats['medium_change'])
        with stat_col5:
            st.metric("High Change", block_stats['high_change'])
        
        st.markdown("---")
        
        # Display images side by side
        st.subheader("Image Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image**")
            if show_grid:
                original_with_grid = create_block_grid_overlay(original, block_size, (0, 255, 0), 1)
                st.image(numpy_to_pil(original_with_grid), use_container_width=True)
            else:
                st.image(numpy_to_pil(original), use_container_width=True)
        
        with col2:
            st.markdown(f"**Compressed (Quality={quality})**")
            if show_grid:
                compressed_highlighted = highlight_changed_blocks(
                    original, reconstructed, block_size, diff_threshold
                )
                st.image(numpy_to_pil(compressed_highlighted), use_container_width=True)
            else:
                st.image(numpy_to_pil(reconstructed), use_container_width=True)
        
        # Legend for highlighted blocks
        if show_grid:
            st.markdown("""
            **Block Highlight Legend:**
            - **Red blocks**: High difference (significant compression artifacts)
            - **Orange blocks**: Medium-high difference
            - **Yellow blocks**: Medium difference
            - **No highlight**: Minimal or no visible change
            """)
        
        # Difference heatmap
        if show_diff_map:
            st.markdown("---")
            st.subheader("Difference Heatmap")
            
            diff_col1, diff_col2 = st.columns(2)
            
            with diff_col1:
                st.markdown("**Pixel-Level Difference**")
                diff_map = create_difference_map(original, reconstructed)
                st.image(numpy_to_pil(diff_map), use_container_width=True)
                st.caption("Blue = No change, Red = High change")
            
            with diff_col2:
                st.markdown("**Changed Blocks Overlay**")
                highlighted = highlight_changed_blocks(original, reconstructed, block_size, diff_threshold)
                st.image(numpy_to_pil(highlighted), use_container_width=True)
                st.caption("Colored blocks indicate areas with compression artifacts")
        
        # Block details
        if show_block_details:
            st.markdown("---")
            st.subheader("Most Changed Blocks (Detailed View)")
            
            # Find most changed blocks
            most_changed = find_most_changed_blocks(original, reconstructed, block_size, 8)
            
            if most_changed:
                st.markdown("**Top 8 blocks with highest compression changes:**")
                
                # Create block comparison grid
                num_cols = 4
                rows = (len(most_changed) + num_cols - 1) // num_cols
                
                for row in range(rows):
                    cols = st.columns(num_cols)
                    for col_idx in range(num_cols):
                        block_idx = row * num_cols + col_idx
                        if block_idx < len(most_changed):
                            x, y = most_changed[block_idx]
                            
                            with cols[col_idx]:
                                # Extract blocks
                                orig_block = original[y:y+block_size, x:x+block_size]
                                comp_block = reconstructed[y:y+block_size, x:x+block_size]
                                
                                # Calculate block MSE
                                block_mse = np.mean((orig_block.astype(np.float32) - 
                                                     comp_block.astype(np.float32)) ** 2)
                                
                                # Scale up for visibility
                                scale = 8
                                orig_scaled = cv2.resize(orig_block, 
                                                         (block_size * scale, block_size * scale),
                                                         interpolation=cv2.INTER_NEAREST)
                                comp_scaled = cv2.resize(comp_block,
                                                         (block_size * scale, block_size * scale),
                                                         interpolation=cv2.INTER_NEAREST)
                                
                                # Create diff visualization
                                diff = np.abs(orig_block.astype(np.float32) - 
                                             comp_block.astype(np.float32))
                                diff = np.clip(diff * 5, 0, 255).astype(np.uint8)
                                diff_scaled = cv2.resize(diff,
                                                         (block_size * scale, block_size * scale),
                                                         interpolation=cv2.INTER_NEAREST)
                                
                                # Combine into one image
                                combined = np.hstack([orig_scaled, comp_scaled, diff_scaled])
                                
                                st.image(numpy_to_pil(combined), use_container_width=True)
                                st.caption(f"Block ({x},{y}) | MSE: {block_mse:.1f}")
                
                st.markdown("*Each block shows: Original | Compressed | Difference (amplified)*")
        
        st.markdown("---")
        
        # Quality comparison
        st.subheader("Quality Level Comparison")
        if st.checkbox("Show comparison at different quality levels"):
            comparison_qualities = [10, 30, 50, 70, 90]
            cols = st.columns(len(comparison_qualities))
            
            for idx, q in enumerate(comparison_qualities):
                compressor.set_quality(q)
                comp_data = compressor.compress(original)
                recon = compressor.decompress(comp_data)
                p = calculate_psnr(original, recon)
                stats = create_block_statistics(original, recon, block_size)
                
                with cols[idx]:
                    st.image(numpy_to_pil(recon), use_container_width=True)
                    st.caption(f"Q={q}")
                    st.caption(f"PSNR={p:.1f}dB")
                    st.caption(f"Changed: {stats['changed_percent']:.0f}%")
        
        # Download section
        st.markdown("---")
        st.subheader("Download")
        
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        with dl_col1:
            # Compressed image
            reconstructed_pil = numpy_to_pil(reconstructed)
            buf = io.BytesIO()
            reconstructed_pil.save(buf, format='JPEG', quality=95)
            buf.seek(0)
            st.download_button(
                label="Download Compressed Image",
                data=buf,
                file_name=f"compressed_q{quality}.jpg",
                mime="image/jpeg"
            )
        
        with dl_col2:
            # Difference map
            diff_map = create_difference_map(original, reconstructed)
            diff_pil = numpy_to_pil(diff_map)
            buf2 = io.BytesIO()
            diff_pil.save(buf2, format='PNG')
            buf2.seek(0)
            st.download_button(
                label="Download Difference Map",
                data=buf2,
                file_name=f"diff_map_q{quality}.png",
                mime="image/png"
            )
        
        with dl_col3:
            # Highlighted blocks image
            highlighted = highlight_changed_blocks(original, reconstructed, block_size, diff_threshold)
            highlighted_pil = numpy_to_pil(highlighted)
            buf3 = io.BytesIO()
            highlighted_pil.save(buf3, format='PNG')
            buf3.seek(0)
            st.download_button(
                label="Download Block Analysis",
                data=buf3,
                file_name=f"block_analysis_q{quality}.png",
                mime="image/png"
            )
    
    else:
        # Show instructions when no image is uploaded
        st.markdown("""
        ### How to use:
        1. Upload an image using the file uploader above
        2. Adjust the quality slider in the sidebar
        3. View the compressed result and quality metrics
        4. Analyze block-level changes with the visualization tools
        
        ### Block Analysis Features:
        - **Block Grid**: Shows 8x8 DCT block boundaries
        - **Changed Block Highlighting**: Colors blocks based on compression impact
        - **Difference Heatmap**: Visualizes pixel-level changes
        - **Block Details**: Zoomed view of most affected blocks
        
        ### About JPEG Compression:
        - **Quality 1-30**: High compression, noticeable artifacts
        - **Quality 31-60**: Balanced compression and quality
        - **Quality 61-90**: Good quality, moderate file size
        - **Quality 91-100**: Near-lossless quality
        
        ### Understanding Block Colors:
        | Color | Meaning |
        |-------|---------|
        | No highlight | Block unchanged or minimal change |
        | Yellow | Medium compression artifacts |
        | Orange | Noticeable compression artifacts |
        | Red | Significant compression artifacts |
        """)


if __name__ == "__main__":
    main()
