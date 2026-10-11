"""Crop table cells from a scanned register page using ruled-line detection.

Stand-in for the GloSAT cell detector while its checkpoints download: the
register is ruled paper, so long horizontal lines give the rows and the
hand-ruled column lines give the columns.

    uv run python grid_cells.py page.png out_dir/
writes out_dir/r{row:02d}_c{col}.png plus out_dir/grid.png (overlay for checking).
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def line_positions(mask, axis, min_gap):
    """Centres of line bands from a binary line mask, projected on one axis."""
    prof = mask.sum(axis=axis).astype(float)
    thr = prof.max() * 0.35
    idx = np.where(prof > thr)[0]
    groups, cur = [], [idx[0]] if len(idx) else []
    for i in idx[1:]:
        if i - cur[-1] <= 3:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    if cur:
        groups.append(cur)
    centres = [int(np.mean(g)) for g in groups]
    merged = []
    for c in centres:
        if merged and c - merged[-1] < min_gap:
            continue
        merged.append(c)
    return merged


def detect_grid(gray):
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    h, w = gray.shape
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 6, 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 6)))
    rows = line_positions(horiz, axis=1, min_gap=40)
    cols = line_positions(vert, axis=0, min_gap=60)
    return rows, cols


def main(page, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(page))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = detect_grid(gray)
    overlay = img.copy()
    for y in rows:
        cv2.line(overlay, (0, y), (img.shape[1], y), (0, 0, 255), 2)
    for x in cols:
        cv2.line(overlay, (x, 0), (x, img.shape[0]), (255, 0, 0), 2)
    cv2.imwrite(str(out / "grid.png"), overlay)
    pad = 4
    for r in range(len(rows) - 1):
        for c in range(len(cols) - 1):
            y0, y1, x0, x1 = rows[r] + pad, rows[r + 1] - pad, cols[c] + pad, cols[c + 1] - pad
            if y1 - y0 > 20 and x1 - x0 > 20:
                cv2.imwrite(str(out / f"r{r:02d}_c{c}.png"), img[y0:y1, x0:x1])
    print(f"{len(rows)} row lines, {len(cols)} column lines -> {out}")
    print("rows:", rows)
    print("cols:", cols)


if __name__ == "__main__":
    main(*sys.argv[1:3])
