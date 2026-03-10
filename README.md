# lichtfeld-node

Windows GPU splat processing node for [MobileScannerPhotogrammetry](https://mobilescannerphotogrammetry.web.app) using [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio).

Runs natively on Windows with real-time training visualization. Polls Firestore for scan tickets, runs COLMAP + LichtFeld Studio training, aligns to ARKit, uploads results.

## Quick Start

### 1. Install Prerequisites

- **NVIDIA Driver** 570+ ([download](https://www.nvidia.com/Download/index.aspx))
- **COLMAP** with GPU support ([download](https://github.com/colmap/colmap/releases))
- **LichtFeld Studio** ([download binary](https://github.com/MrNeRF/LichtFeld-Studio/releases)) — unzip to `C:\tools\LichtFeld-Studio\`
- **Python 3.10+** ([download](https://www.python.org/downloads/))

Make sure `colmap` is on your PATH, or pass `--colmap "C:\path\to\colmap.exe"`.

### 2. Clone & Setup

```bash
git clone https://github.com/theLodgeBots/lichtfeld-node.git
cd lichtfeld-node
```

Copy `service-account.json` from the MobileScannerPhotogrammetry project into this directory.

Double-click **`setup.bat`** — this creates a Python virtual environment and installs all dependencies.

### 3. Run

Double-click **`run.bat`** or:

```bash
.venv\Scripts\activate
python lichtfeld_node.py --lfs "C:\tools\LichtFeld-Studio\bin\LichtFeld-Studio.exe"
```

The LichtFeld Studio viewer window will open during training so you can watch gaussians form in real time.

Dashboard at http://localhost:8788

## What It Does

1. Watches Firestore for `splat_status=300` (requested) tickets
2. Downloads dataset from Firebase Storage
3. Converts HEIC → JPG (if needed)
4. Runs COLMAP SfM (GPU-accelerated SIFT, exhaustive matching, undistortion)
5. Trains gaussian splats using LichtFeld Studio with MCMC + bilateral grid + mip filtering
6. Aligns to ARKit coordinate space (Procrustes)
7. Exports PLY → .splat format
8. Uploads to Firebase Storage
9. Updates ticket status to 500 (success)

## Configuration

```
--lfs PATH          Path to LichtFeld-Studio executable (required)
--port 8788         Dashboard port
--data-dir ./data   Working directory
--max-steps 30000   Training iterations
--downsample 1      Image downsample factor (1 = full res)
--strategy mcmc     Training strategy: mcmc (default) or default (ADC)
--colmap colmap     Path to COLMAP executable
--headless          Disable LichtFeld viewer (run without GUI)
```

You can pass extra args through `run.bat`, e.g.:
```
run.bat --max-steps 50000 --headless
```

## LichtFeld Studio Features Used

- **MCMC strategy** — better gaussian placement with fixed budget
- **Bilateral grid** — per-image appearance/exposure correction
- **Mip filtering** — anti-aliasing for multi-scale viewing
- **Real-time viewer** — watch training progress live (disable with `--headless`)

## Comparison with gsplat-node

| Feature | gsplat-node | lichtfeld-node |
|---|---|---|
| Platform | Docker (Linux) | Native Windows |
| Trainer | gsplat (Python/CUDA) | LichtFeld Studio (C++/CUDA) |
| Rasterization | gsplat | 2.4x faster custom CUDA |
| Real-time viewer | No | Yes |
| Strategy | default or mcmc | mcmc (default) or ADC |
| Appearance model | bilateral grid | bilateral grid |
| License | Apache 2.0 | GPLv3 |

## Requirements

- Windows 10/11
- NVIDIA GPU with compute capability 7.5+ (RTX 2060+, tested: RTX 4080)
- NVIDIA driver 570+
- 8GB+ VRAM recommended

## License

GPLv3 (due to LichtFeld Studio dependency)
