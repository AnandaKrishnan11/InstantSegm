import numpy as np
import rasterio as rio

def _tapered_weights(h, w):
    wy = np.hanning(h + 2)[1:-1]
    wx = np.hanning(w + 2)[1:-1]
    return np.outer(wy, wx).astype(np.float32)

def _offsets(total, chip_size, stride):
    if total <= chip_size:
        return [0]
    offs = list(range(0, total - chip_size + 1, stride))
    if offs[-1] != total - chip_size:
        offs.append(total - chip_size)
    return offs

def chipImage(img_path, chip_size=512, stride=None, mask_path=None):
    stride = chip_size // 2 if stride is None else stride

    src = rio.open(img_path)
    profile = src.profile

    if mask_path is not None:
        msrc = rio.open(mask_path)
        masks = []

    chips, windows = [], []

    for row in _offsets(src.height, chip_size, stride):
        for col in _offsets(src.width, chip_size, stride):
            win = rio.windows.Window(col, row, chip_size, chip_size)
            chips.append(src.read(window=win, boundless=True, fill_value=0))
            windows.append((row, col))
            if mask_path is not None:
                masks.append(msrc.read(window=win, boundless=True, fill_value=0))

    meta = {"profile": profile,
            "height": src.height,
            "width": src.width,
            "chip_size": chip_size,
            "stride": stride,
            "windows": windows}

    src.close()
    if mask_path is not None:
        msrc.close()
        return chips, masks, meta
    return chips, meta

def mosaicChips(preds, meta, save_path):
 
    profile = meta["profile"]
    H, W = meta["height"], meta["width"]
    windows = meta["windows"]

    out_dtype = preds[0].dtype
    bands = preds[0].shape[0] if preds[0].ndim == 3 else 1

    acc = np.zeros((bands, H, W), dtype=np.float32)
    wsum = np.zeros((1, H, W), dtype=np.float32)

    for pred, (row, col) in zip(preds, windows):
        pred = pred if pred.ndim == 3 else pred[None]
        h = min(pred.shape[1], H - row)
        w = min(pred.shape[2], W - col)

        weight = _tapered_weights(pred.shape[1], pred.shape[2])[:h, :w]

        acc[:, row:row+h, col:col+w] += pred[:, :h, :w].astype(np.float32) * weight
        wsum[:, row:row+h, col:col+w] += weight

    canvas = np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0)

    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        canvas = np.clip(np.rint(canvas), info.min, info.max)
    canvas = canvas.astype(out_dtype)

    out_meta = profile.copy()
    out_meta.update({"height": H, "width": W, "count": bands, "dtype": out_dtype})
    with rio.open(save_path, "w", **out_meta) as dst:
        dst.write(canvas)
    print(f"Mosaic image is saved in {save_path}")
