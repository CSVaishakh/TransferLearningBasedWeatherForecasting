"""Clean a cell crop for OCR: remove ruled lines, crop tightly to the ink,
and pad with white so the value sits like a text line (what TrOCR expects)."""
import cv2
import numpy as np
from PIL import Image


def clean_cell(img: Image.Image, pad: int = 12) -> Image.Image:
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    h, w = bw.shape
    # ruled lines: horizontal runs wider than ~60% of the crop, vertical runs taller than ~70%
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * 0.6)), 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * 0.7)))))
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    ink = cv2.bitwise_and(bw, cv2.bitwise_not(lines))
    # drop tiny specks but keep decimal dots (small, roughly round blobs)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 12:
            keep[lab == i] = 255
    ys, xs = np.nonzero(keep)
    if len(xs) == 0:
        return img
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    # render ink from the original grey values (keeps stroke shape), on white
    out = np.full_like(gray, 255)
    out[keep > 0] = gray[keep > 0]
    out = out[y0:y1 + 1, x0:x1 + 1]
    out = cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return Image.fromarray(out).convert("RGB")
