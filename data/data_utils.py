import os
import numpy as np
from glob import glob
import rasterio as rio
from rasterio.merge import merge


def chipImage(img_path, chip_size=512, mask_path=None):
    src = rio.open(img_path)
    profile = src.profile

    if mask_path is not None:
        msrc = rio.open(mask_path)
        masks = []

    chips = []
    windows = []

    for row in range(0, src.height, chip_size):
        for col in range(0, src.width, chip_size):
            win = rio.windows.Window(col, row, chip_size, chip_size)
            chip = src.read(window=win, boundless=True, fill_value=0)
            chips.append(chip)
            windows.append((row, col))

            if mask_path is not None:
                mchip = msrc.read(window=win, boundless=True, fill_value=0)
                masks.append(mchip)

    meta = {"profile": profile,
            "height": src.height,
            "width": src.width,
            "windows": windows}
    
    src.close()
    if mask_path is not None:
        return chips, masks, meta
    return chips, meta


def mosaicChips(preds, meta, save_path):
    profile = meta["profile"]
    H, W = meta["height"], meta["width"]
    windows = meta["windows"]

    bands = preds[0].shape[0] if preds[0].ndim == 3 else 1
    canvas = np.zeros((bands, H, W), dtype=preds[0].dtype)

    for pred, (row, col) in zip(preds, windows):
        pred = pred if pred.ndim == 3 else pred[None]
        h = min(pred.shape[1], H - row)
        w = min(pred.shape[2], W - col)
        canvas[:, row:row+h, col:col+w] = pred[:, :h, :w]

    out_meta = profile.copy()
    out_meta.update({"height": H,
                     "width": W,
                     "count": bands,
                     "dtype": canvas.dtype})
    with rio.open(save_path, "w", **out_meta) as dst:
        dst.write(canvas)
    print(f"Mosaic image is saved in {save_path}")
