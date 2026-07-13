# InstantSegm: SAR Surface-Water Segmentation 

**Author:** Ananda Krishnan
*Geospatial Python — Final Project*

Turn a raw Sentinel-1 SAR scene into a clean, georeferenced surface-water map, packaged as a Docker image so it reproduces on any machine with a single command.

---

## What this does

Optical satellites are blind under cloud — exactly the conditions common in mountainous and temperate regions for much of the year. Synthetic Aperture Radar (Sentinel-1) sees through cloud, day or night, which makes it a reliable source for monitoring surface-water extent, one of the most direct indicators of how much water a region has on hand.

The pipeline takes a single Sentinel-1 scene and:

1. **Chips** it into 224×224 tiles.
2. **Segments** each tile with a trained deep-learning model, labelling every pixel as *water* or *not-water*.
3. **Stitches** the predictions back into one georeferenced raster covering the original footprint, preserving the geotransform.

The output is a water mask (`.tif`) from which total water-covered area can be read off.

## Model

A DeepLab-based semantic segmentation network, trained on the **Sen1Floods11** dataset (Sentinel-1 imagery with hand-labelled water masks). The trained weights are **baked into the Docker image**, so the container only ever does **inference** — it loads a fixed model and runs a new scene through it. This keeps the container light and the results fully reproducible.

Inference runs on **CPU** — no GPU required.

## Tech stack

- **PyTorch** (CPU build) — the segmentation model
- **rasterio** — tiling, stitching, and geotransform handling
- **Docker** — packaging for reproducibility

---

## Project structure

```
Deeplab/
├── checkpoint/
│   └── best_weight.pt      # trained weights (baked into the image)
├── data/
│   ├── __init__.py
│   ├── data_utils.py       # chipping / mosaicking
│   └── loader.py           # dataset + loader
├── engine.py               # inference loop
├── model.py                # model definition
├── test.py                 # entry point (inference CLI)
├── utils.py                # metrics / helpers
├── requirements.txt        # pinned dependencies
├── Dockerfile              # build recipe
└── .dockerignore
```

---

## Prerequisites

You need **Docker** installed. On Windows, the cleanest route is Docker Engine inside a WSL2 Ubuntu distribution (no Docker Desktop required):

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

> On WSL2 the Docker daemon does not auto-start. If you ever see *"Cannot connect to the Docker daemon"*, run `sudo service docker start` again.

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

The container is sealed off from your filesystem, so you connect an **input** folder and an **output** folder using `-v` volume mounts. Using `$(pwd)` (your current folder) keeps the command working on any machine — no absolute paths to edit.

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
  --source_path /data/input \
  --save_path /data/output/water_map.tif
```

The water map appears at `./output/water_map.tif`.

**How it works:**

- `--rm` deletes the container after it finishes.
- `-v host:container` mounts a folder from your machine into the container. The **left** side is your real path; the **right** side (`/data/input`, `/data/output`) is where it appears inside the container.
- Arguments after `waterseg` are passed to `test.py`. They use the **container-side** paths (`/data/...`), because the code runs inside the container.

### Windows PowerShell note

The `$(pwd)` syntax is Bash (WSL / Linux / macOS). The simplest approach is to run everything **inside WSL**. If you must run from PowerShell, use `${PWD}`:

```powershell
docker run --rm -v "${PWD}/input:/data/input" -v "${PWD}/output:/data/output" waterseg --source_path /data/input --save_path /data/output/water_map.tif
```

---

## Reproducibility notes

- The container uses **Python 3.12** internally, independent of whatever Python is on the host.
- PyTorch and torchvision are pinned to the **CPU** build to keep the image small.
- Dependencies are pinned in `requirements.txt`.
- The trained weights are committed inside the image, so no external download is needed at run time.

## Notes

Generative AI was used for debugging and for the containerisation setup; the design and analysis are the author's own.
