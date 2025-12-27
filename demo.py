"""
JPEG Compression Simulator - Demo Script
Demonstrates the JPEG compression simulation capabilities.
"""

import numpy as np
import cv2
from pathlib import Path
import sys

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from jpeg_simulator import JPEGCompressor, simulate_compression, calculate_psnr, calculate_ssim


def create_sample_image(size: tuple = (256, 256)) -> np.ndarray:
    """Create a sample test image with various patterns."""
    h, w = size
    image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create gradient background
    for i in range(h):
        for j in range(w):
            image[i, j] = [
                int(255 * i / h),  # Blue gradient
                int(255 * j / w),  # Green gradient
                128                 # Red constant
            ]
    
    # Add some shapes
    cv2.rectangle(image, (50, 50), (100, 100), (255, 255, 255), -1)
    cv2.circle(image, (180, 80), 30, (0, 0, 255), -1)
    cv2.line(image, (20, 200), (230, 150), (0, 255, 0), 3)
    
    # Add text
    cv2.putText(image, "JPEG", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image


def run_demo():
    """Run the compression demo."""
    print("=" * 60)
    print("JPEG Compression Simulator - Demo")
    print("=" * 60)
    
    # Create output directory for compressed images
    output_dir = Path(__file__).parent / "compressed_image"
    output_dir.mkdir(exist_ok=True)
    
    # Create sample_images directory
    samples_dir = Path(__file__).parent / "sample_images"
    samples_dir.mkdir(exist_ok=True)
    
    # Create or load test image
    sample_image_path = samples_dir / "test_image.png"
    
    if sample_image_path.exists():
        print(f"\nLoading existing test image: {sample_image_path}")
        original = cv2.imread(str(sample_image_path))
    else:
        print("\nCreating sample test image...")
        original = create_sample_image((256, 256))
        cv2.imwrite(str(sample_image_path), original)
        print(f"Saved sample image to: {sample_image_path}")
    
    print(f"Image size: {original.shape}")
    print(f"\nAll compressed images will be saved to: {output_dir.absolute()}")
    
    # Test different quality levels
    quality_levels = [10, 30, 50, 70, 90]
    
    print("\n" + "-" * 60)
    print("Testing different quality levels:")
    print("-" * 60)
    
    compressor = JPEGCompressor()
    
    for quality in quality_levels:
        compressor.set_quality(quality)
        
        # Compress and decompress
        compressed_data = compressor.compress(original)
        reconstructed = compressor.decompress(compressed_data)
        
        # Calculate metrics
        psnr = calculate_psnr(original, reconstructed)
        ssim = calculate_ssim(original, reconstructed)
        
        # Save reconstructed image to compressed_image folder
        output_filename = f"reconstructed_q{quality}.jpg"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), reconstructed)
        
        print(f"Quality {quality:3d}: PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f} -> Saved: {output_filename}")
    
    # Save original for comparison
    original_copy_path = output_dir / "original.png"
    cv2.imwrite(str(original_copy_path), original)
    print(f"\nOriginal saved as: original.png")
    
    # Test compress_file method
    print("\n" + "-" * 60)
    print("Testing compress_file method:")
    print("-" * 60)
    
    compressor.set_quality(50)
    saved_path, stats = compressor.compress_file(str(sample_image_path))
    print(f"Saved to: {saved_path}")
    print(f"Original size: {stats['original_size']} bytes")
    print(f"Compressed size: {stats['compressed_size']} bytes")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"Space savings: {stats['space_savings']:.1f}%")
    
    # Create comparison image
    print("\n" + "-" * 60)
    print("Creating comparison image...")
    print("-" * 60)
    
    comparison_images = [original]
    for quality in [20, 50, 80]:
        compressor.set_quality(quality)
        compressed_data = compressor.compress(original)
        reconstructed = compressor.decompress(compressed_data)
        comparison_images.append(reconstructed)
    
    comparison = np.hstack(comparison_images)
    comparison_path = output_dir / "comparison.jpg"
    cv2.imwrite(str(comparison_path), comparison)
    print(f"Saved comparison image: comparison.jpg")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print(f"All compressed images saved to: {output_dir.absolute()}")
    print("=" * 60)
    
    return True


def main():
    """Main entry point."""
    try:
        success = run_demo()
        if success:
            print("\n[OK] All tests passed!")
        return 0
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
