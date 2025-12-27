#!/bin/bash
# JPEG Compression Simulator - Easy Launcher for Linux/Mac

echo "========================================"
echo "JPEG Compression Simulator"
echo "========================================"
echo ""
echo "Choose an option:"
echo ""
echo "1. Run Demo (Generate test images)"
echo "2. Launch Streamlit Web App"
echo "3. Launch Flask Web App"
echo "4. Launch PyQt6 Desktop App"
echo "5. Install Dependencies"
echo "6. Exit"
echo ""
read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "Running demo script..."
        python demo.py
        ;;
    2)
        echo ""
        echo "Launching Streamlit..."
        echo "Press Ctrl+C to stop the server"
        streamlit run app_streamlit.py
        ;;
    3)
        echo ""
        echo "Launching Flask..."
        echo "Open browser to: http://localhost:5000"
        echo "Press Ctrl+C to stop the server"
        python app_flask.py
        ;;
    4)
        echo ""
        echo "Launching PyQt6 Desktop App..."
        python app_pyqt6.py
        ;;
    5)
        echo ""
        echo "Installing dependencies..."
        pip install -r requirements.txt
        echo ""
        echo "Installation complete!"
        ;;
    6)
        echo ""
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice!"
        ;;
esac
