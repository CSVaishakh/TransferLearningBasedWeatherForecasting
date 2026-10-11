"""Crop individual handwritten values from a register page for OCR testing.

Ruled lines are removed, remaining ink is grouped into value-sized blobs, and
each blob's bounding box (padded) is cropped from the original page. This is
independent of table-structure detection, so OCR can be evaluated on its own.

    uv run python value_crops.py page.png out_dir/ [--top PX]
writes out_dir/v{id:03d}.png, out_dir/boxes.csv and out_dir/boxes.png.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("out")
    ap.add_argument("--top", type=int, default=0, help="ignore ink above this y (title/header)")
    ap.add_argument("--bottom", type=int, default=0, help="ignore ink below this y (0 = none)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(a.page)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 18)

    # ruled lines: long thin horizontal / vertical runs
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (120, 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 120)))
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    ink = cv2.bitwise_and(bw, cv2.bitwise_not(lines))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # speckle

    # merge characters of one value: wide horizontal dilation, little vertical
    blobs = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(blobs)
    boxes = []
    for i in range(1, n):
        x, y, bw_, bh, area = stats[i]
        if bh < 18 or bw_ < 18 or bh > 140 or bw_ > 420:
            continue
        if y < a.top or (a.bottom and y > a.bottom):
            continue
        boxes.append((x, y, bw_, bh))
    # reading order: row bands, then left to right
    boxes.sort(key=lambda b: (round((b[1] + b[3] / 2) / 60), b[0]))

    vis = img.copy()
    with open(out / "boxes.csv", "w") as f:
        f.write("id,x,y,w,h\n")
        for k, (x, y, bw_, bh) in enumerate(boxes):
            p = 6
            crop = img[max(0, y - p): y + bh + p, max(0, x - p): x + bw_ + p]
            cv2.imwrite(str(out / f"v{k:03d}.png"), crop)
            f.write(f"{k},{x},{y},{bw_},{bh}\n")
            cv2.rectangle(vis, (x, y), (x + bw_, y + bh), (0, 0, 255), 2)
            cv2.putText(vis, str(k), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite(str(out / "boxes.png"), vis)
    print(f"{len(boxes)} value crops -> {out}")


if __name__ == "__main__":
    main()
