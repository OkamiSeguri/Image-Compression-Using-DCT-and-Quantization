"""
JPEG Compression Simulator - PyQt6 Desktop Application
A desktop GUI application for JPEG compression simulation.
"""

import sys
import platform
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QGroupBox, QMessageBox,
    QStatusBar, QCheckBox, QSpinBox, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QFont

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from jpeg_simulator import JPEGCompressor, calculate_psnr, calculate_ssim

# Create compressed_image folder
COMPRESSED_FOLDER = Path(__file__).parent / "compressed_image"
COMPRESSED_FOLDER.mkdir(exist_ok=True)


class JPEGCompressorApp(QMainWindow):
    """Main application window for JPEG compression simulator."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JPEG Compression Simulator")
        self.setMinimumSize(1100, 700)
        
        self.original_image = None
        self.compressed_image = None
        self.compressor = JPEGCompressor()
        self.current_filename = "image"
        
        self.init_ui()
        self.update_status()
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("JPEG Compression Simulator")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        # Load button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setMinimumWidth(120)
        controls_layout.addWidget(self.load_btn)
        
        # Quality control
        controls_layout.addWidget(QLabel("Quality:"))
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(50)
        self.quality_slider.setMinimumWidth(200)
        self.quality_slider.valueChanged.connect(self.on_quality_changed)
        controls_layout.addWidget(self.quality_slider)
        
        self.quality_spinbox = QSpinBox()
        self.quality_spinbox.setRange(1, 100)
        self.quality_spinbox.setValue(50)
        self.quality_spinbox.valueChanged.connect(self.on_spinbox_changed)
        controls_layout.addWidget(self.quality_spinbox)
        
        # Auto-save checkbox
        self.auto_save_cb = QCheckBox("Auto-save")
        self.auto_save_cb.setChecked(True)
        self.auto_save_cb.setToolTip("Automatically save compressed images")
        controls_layout.addWidget(self.auto_save_cb)
        
        # Compress button
        self.compress_btn = QPushButton("Compress")
        self.compress_btn.clicked.connect(self.compress_image)
        self.compress_btn.setEnabled(False)
        self.compress_btn.setMinimumWidth(100)
        controls_layout.addWidget(self.compress_btn)
        
        # Save button
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_compressed)
        self.save_btn.setEnabled(False)
        self.save_btn.setMinimumWidth(80)
        controls_layout.addWidget(self.save_btn)
        
        # Open folder button
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        self.open_folder_btn.setMinimumWidth(100)
        controls_layout.addWidget(self.open_folder_btn)
        
        main_layout.addWidget(controls_group)
        
        # Image display area
        images_layout = QHBoxLayout()
        
        # Original image group
        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout(original_group)
        self.original_label = QLabel("Load an image to start")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(450, 400)
        self.original_label.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self.original_label.setStyleSheet("background-color: #f0f0f0;")
        original_layout.addWidget(self.original_label)
        self.original_info = QLabel("No image loaded")
        self.original_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        original_layout.addWidget(self.original_info)
        images_layout.addWidget(original_group)
        
        # Compressed image group
        compressed_group = QGroupBox("Compressed Image")
        compressed_layout = QVBoxLayout(compressed_group)
        self.compressed_label = QLabel("Compressed result will appear here")
        self.compressed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.compressed_label.setMinimumSize(450, 400)
        self.compressed_label.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self.compressed_label.setStyleSheet("background-color: #f0f0f0;")
        compressed_layout.addWidget(self.compressed_label)
        self.compressed_info = QLabel("No compression yet")
        self.compressed_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        compressed_layout.addWidget(self.compressed_info)
        images_layout.addWidget(compressed_group)
        
        main_layout.addLayout(images_layout)
        
        # Metrics group
        metrics_group = QGroupBox("Quality Metrics")
        metrics_layout = QHBoxLayout(metrics_group)
        
        self.psnr_label = QLabel("PSNR: -- dB")
        self.psnr_label.setFont(QFont("Arial", 11))
        metrics_layout.addWidget(self.psnr_label)
        
        self.ssim_label = QLabel("SSIM: --")
        self.ssim_label.setFont(QFont("Arial", 11))
        metrics_layout.addWidget(self.ssim_label)
        
        metrics_layout.addStretch()
        
        self.folder_label = QLabel(f"Output folder: {COMPRESSED_FOLDER}")
        self.folder_label.setFont(QFont("Arial", 9))
        metrics_layout.addWidget(self.folder_label)
        
        main_layout.addWidget(metrics_group)
        
        # Status bar
        self.setStatusBar(QStatusBar())
    
    def update_status(self):
        """Update status bar with file count."""
        # Count multiple image formats
        file_count = 0
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            file_count += len(list(COMPRESSED_FOLDER.glob(ext)))
        self.statusBar().showMessage(f"Saved files: {file_count} | Output: {COMPRESSED_FOLDER.absolute()}")
    
    def on_quality_changed(self, value):
        """Handle quality slider change."""
        self.quality_spinbox.blockSignals(True)
        self.quality_spinbox.setValue(value)
        self.quality_spinbox.blockSignals(False)
        
        if self.original_image is not None and self.auto_save_cb.isChecked():
            self.compress_image()
    
    def on_spinbox_changed(self, value):
        """Handle quality spinbox change."""
        self.quality_slider.blockSignals(True)
        self.quality_slider.setValue(value)
        self.quality_slider.blockSignals(False)
        
        if self.original_image is not None and self.auto_save_cb.isChecked():
            self.compress_image()
    
    def load_image(self):
        """Load an image from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image", 
            "", 
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff);;All Files (*)"
        )
        if file_path:
            # Use IMREAD_COLOR to ensure 3-channel BGR image
            self.original_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.current_filename = Path(file_path).stem
            
            if self.original_image is not None:
                self.display_image(self.original_image, self.original_label)
                h, w = self.original_image.shape[:2]
                file_size = Path(file_path).stat().st_size
                size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.2f} MB"
                self.original_info.setText(f"Size: {w}x{h} | {size_str} | {Path(file_path).name}")
                self.compress_btn.setEnabled(True)
                self.statusBar().showMessage(f"Loaded: {file_path}")
            else:
                QMessageBox.warning(self, "Error", f"Could not load image: {file_path}\n\nMake sure the file exists and is a valid image format.")
    
    def compress_image(self):
        """Compress the loaded image."""
        if self.original_image is None:
            return
        
        quality = self.quality_slider.value()
        self.compressor.set_quality(quality)
        
        # Compress and decompress
        compressed_data = self.compressor.compress(self.original_image)
        self.compressed_image = self.compressor.decompress(compressed_data)
        
        # Calculate metrics
        psnr = calculate_psnr(self.original_image, self.compressed_image)
        ssim = calculate_ssim(self.original_image, self.compressed_image)
        
        # Update UI
        self.psnr_label.setText(f"PSNR: {psnr:.2f} dB")
        self.ssim_label.setText(f"SSIM: {ssim:.4f}")
        
        self.display_image(self.compressed_image, self.compressed_label)
        self.compressed_info.setText(f"Quality: {quality}")
        self.save_btn.setEnabled(True)
        
        # Auto-save if enabled
        if self.auto_save_cb.isChecked():
            saved_path = self.save_to_folder()
            self.statusBar().showMessage(f"Compressed and saved: {Path(saved_path).name}")
            self.update_status()
    
    def save_to_folder(self) -> str:
        """Save compressed image to compressed_image folder."""
        if self.compressed_image is None:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quality = self.quality_slider.value()
        filename = f"{self.current_filename}_q{quality}_{timestamp}.jpg"
        output_path = COMPRESSED_FOLDER / filename
        cv2.imwrite(str(output_path), self.compressed_image)
        return str(output_path)
    
    def save_compressed(self):
        """Manually save the compressed image."""
        if self.compressed_image is None:
            return
        
        saved_path = self.save_to_folder()
        if saved_path:
            self.update_status()
            QMessageBox.information(
                self, 
                "Image Saved", 
                f"Compressed image saved to:\n{saved_path}"
            )
    
    def open_output_folder(self):
        """Open the output folder in file explorer (cross-platform)."""
        import subprocess
        
        folder_path = str(COMPRESSED_FOLDER.absolute())
        system = platform.system()
        
        try:
            if system == 'Windows':
                subprocess.Popen(f'explorer "{folder_path}"', shell=True)
            elif system == 'Darwin':  # macOS
                subprocess.Popen(['open', folder_path])
            else:  # Linux and others
                subprocess.Popen(['xdg-open', folder_path])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder: {e}")
    
    def display_image(self, image: np.ndarray, label: QLabel):
        """Display an image on a QLabel."""
        try:
            # Handle grayscale images
            if len(image.shape) == 2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            # Make a copy to ensure data persistence
            rgb_image = np.ascontiguousarray(rgb_image)
            
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            # Copy the image data to avoid memory issues
            qt_image = qt_image.copy()
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
        except Exception as e:
            label.setText(f"Error displaying image: {e}")


def main():
    """Main entry point for the application."""
    print("\n" + "=" * 60)
    print("JPEG Compression Simulator - PyQt6 Desktop Application")
    print("=" * 60)
    print(f"Compressed images will be saved to: {COMPRESSED_FOLDER.absolute()}")
    print("=" * 60 + "\n")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = JPEGCompressorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
