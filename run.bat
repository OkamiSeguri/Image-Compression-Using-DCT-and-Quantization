@echo off
REM JPEG Compression Simulator - Easy Launcher for Windows

echo ========================================
echo JPEG Compression Simulator
echo ========================================
echo.
echo Choose an option:
echo.
echo 1. Run Demo (Generate test images)
echo 2. Launch Streamlit Web App
echo 3. Launch Flask Web App
echo 4. Launch PyQt6 Desktop App
echo 5. Install Dependencies
echo 6. Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto demo
if "%choice%"=="2" goto streamlit
if "%choice%"=="3" goto flask
if "%choice%"=="4" goto pyqt6
if "%choice%"=="5" goto install
if "%choice%"=="6" goto end

echo Invalid choice!
pause
goto end

:demo
echo.
echo Running demo script...
python demo.py
pause
goto end

:streamlit
echo.
echo Launching Streamlit...
echo Press Ctrl+C to stop the server
streamlit run app_streamlit.py
pause
goto end

:flask
echo.
echo Launching Flask...
echo Open browser to: http://localhost:5000
echo Press Ctrl+C to stop the server
python app_flask.py
pause
goto end

:pyqt6
echo.
echo Launching PyQt6 Desktop App...
python app_pyqt6.py
pause
goto end

:install
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Installation complete!
pause
goto end

:end
echo.
echo Goodbye!
