"""Make the GloSAT inference scripts run without a GPU.

1. Device strings: 'cuda:0' -> GLOSAT_DEVICE env var (default 'cpu').
2. mmcv 0.4.3's scatter() only moves data to GPUs, so on CPU we unwrap the
   collated DataContainers by hand, as later mmdetection versions do.
"""
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])

for f in list((root / "dla" / "src").glob("inference*.py")) + [root / "mmdet" / "apis" / "inference.py"]:
    src = f.read_text()
    new = src.replace("device='cuda:0'", "device=__import__('os').environ.get('GLOSAT_DEVICE', 'cpu')")
    if new != src:
        f.write_text(new)
        print("device patched:", f.relative_to(root))

api = root / "mmdet" / "apis" / "inference.py"
src = api.read_text()
old = "    data = scatter(collate([data], samples_per_gpu=1), [device])[0]\n"
new = (
    "    data = collate([data], samples_per_gpu=1)\n"
    "    if next(model.parameters()).is_cuda:\n"
    "        data = scatter(data, [device])[0]\n"
    "    else:\n"
    "        key = 'img_metas' if 'img_metas' in data else 'img_meta'\n"
    "        data[key] = [dc.data[0] for dc in data[key]]\n"
)
n = src.count(old)
if n == 0:
    sys.exit("scatter line not found in mmdet/apis/inference.py - upstream changed?")
api.write_text(src.replace(old, new))
print(f"scatter patched in {n} place(s):", api.relative_to(root))
