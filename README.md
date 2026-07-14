# InstantSegm: Surface-Water Segmentation 
---

## What this does

The pipeline takes a single Sentinel-1 scene and:

1. **Chips** it into 512x512 tiles.
2. **Segments** each tile with a trained DeepLab v3 model, labelling every pixel as water or not-water.
3. **Stitches** the predictions back into one georeferenced raster covering the original footprint, preserving the geotransform.

The output is a water mask (`.tif`) from which total water-covered area can be read off.

## Model

A DeepLab v3 semantic segmentation network, trained on the **Sen1Floods11** dataset. The trained weights are baked into the Docker image, so the container only ever does inference it loads a fixed model and runs a new scene through it.

Inference runs on **CPU** no GPU required.

## Tech stack

- **PyTorch** (CPU build) — the segmentation model
- **rasterio** — tiling, stitching, and geotransform handling
- **Docker** — packaging for reproducibility

---

## Project structure

```
Deeplab/
├── checkpoint/
│   └── best_weight.pt      
├── data/
│   ├── __init__.py
│   ├── data_utils.py
│   └── loader.py          
├── engine.py               
├── model.py               
├── test.py                 
├── utils.py                
├── requirements.txt       
├── Dockerfile            
└── .dockerignore
```

---
> ⚠️ **Model weights required before building.**
> The trained weights (`best_weight.pt`) are too large for this repository and are hosted separately.
> **Download them here: [ADD YOUR LINK]** and place the file at `checkpoint/best_weight.pt`
> **before** running `docker build`. The build bakes the weights into the image, so they must
> be present at build time — otherwise inference will fail at run time with a missing-file error.

## Prerequisites

You need Docker installed. On Windows, the cleanest route is Docker Engine inside a WSL2 Ubuntu distribution (no Docker Desktop required):

```bash
# inside a WSL2 Ubuntu shell
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER   # then close and reopen the shell
```

Start the daemon and verify:

```bash
sudo service docker start
docker run hello-world
```
On Linux/macOS, install Docker the usual way for your platform.

---

## Build

From the project root (the folder containing the `Dockerfile`):

```bash
docker build -t waterseg .
```

- `-t waterseg` names the image `waterseg`.
- The trailing `.` sets the build context to the current folder.

The first build takes a few minutes (it downloads Python, PyTorch, and rasterio). Later builds reuse cached layers and are much faster.

---

## Run

The container is sealed off from your filesystem, so you connect an input folder and an output folder using `-v` volume mounts. Using `$(pwd)` (your current folder) keeps the command working on any machine no absolute paths to edit.

From the project root:

```bash
# 1. place a Sentinel-1 scene in ./input
# 2. make an output folder
mkdir -p output

# 3. run
docker run --rm \
  -v "$(pwd)/input":/data/input \
  -v "$(pwd)/output":/data/output \
  waterseg \
  --source_path /data/input/img.tif \  #direct path of the scene
  --save_path /data/output/water_mask.tif #output path with .tif extension
```

The water map appears at `./output/water_mask.tif`.

### Windows PowerShell note

The `$(pwd)` syntax is Bash (WSL / Linux / macOS). The simplest approach is to run everything inside WSL. If you must run from PowerShell, use `${PWD}`:

```powershell
docker run --rm -v "${PWD}/input:/data/input" -v "${PWD}/output:/data/output" waterseg --source_path /data/input --save_path /data/output/
```

---

## Reproducibility notes

- The container uses Python 3.12 internally, independent of whatever Python is on the host.
- PyTorch and torchvision are pinned to the **CPU** build to keep the image small.
- Dependencies are pinned in `requirements.txt`.
- The trained weights are committed inside the image, so no external download is needed at run time.

## Notes

Generative AI was used for debugging;
