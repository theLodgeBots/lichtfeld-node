@echo off
echo === lichtfeld-node setup ===
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check COLMAP
colmap -h >nul 2>&1
if errorlevel 1 (
    echo WARNING: COLMAP not found on PATH.
    echo Download from https://github.com/colmap/colmap/releases
    echo Or pass --colmap "C:\path\to\colmap.exe" when running.
    echo.
)

REM Check nvidia-smi
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvidia-smi not found. Install NVIDIA driver 570+.
    pause
    exit /b 1
)

REM Install deps
echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo === Setup complete ===
echo.
echo Next steps:
echo 1. Place service-account.json in this directory
echo 2. Download LichtFeld Studio from https://github.com/MrNeRF/LichtFeld-Studio/releases
echo 3. Run: python lichtfeld_node.py --lfs "C:\path\to\LichtFeld-Studio.exe"
echo.
pause
