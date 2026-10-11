"""Zero-shot TrOCR on hand-labelled register cells, with and without a
digit-only vocabulary restriction.

    uv run python ocr_test.py [--models small base] [--out eval/results]

Reads eval/sample_cells/labels.csv, writes per-cell predictions and prints
exact-match accuracy and character error rate (CER) by model, mode and kind.
"""
import argparse
import csv
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

HERE = Path(__file__).parent
MODELS = {
    "small": "microsoft/trocr-small-handwritten",
    "base": "microsoft/trocr-base-handwritten",
}
ALLOWED_CHARS = set("0123456789.:")


def normalise(s):
    """Map the register's decimal marks to '.', drop spaces."""
    return s.replace("·", ".").replace(",", ".").replace(" ", "").strip()


def cer(pred, ref):
    """Character error rate via Levenshtein distance."""
    d = list(range(len(ref) + 1))
    for i, pc in enumerate(pred, 1):
        prev, d[0] = d[0], i
        for j, rc in enumerate(ref, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (pc != rc))
            prev, d[j] = d[j], cur
    return d[len(ref)] / max(1, len(ref))


def digit_token_ids(tokenizer):
    """(first, rest): token ids made only of digits / '.' / ':'.
    `first` may carry a leading space (word start); `rest` may not, so the
    decoder cannot start a second "word" such as ' 000 000 ...'."""
    first, rest = [], []
    for tok, idx in tokenizer.get_vocab().items():
        text = tokenizer.convert_tokens_to_string([tok])
        core = text.lstrip(" ")
        if not core or not set(core) <= ALLOWED_CHARS:
            continue
        first.append(idx)
        if core == text:
            rest.append(idx)
    end = [tokenizer.eos_token_id]
    return sorted(set(first)), sorted(set(rest + end))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["small", "base"], choices=list(MODELS))
    ap.add_argument("--cells", default=HERE / "eval" / "sample_cells", type=Path)
    ap.add_argument("--out", default=HERE / "eval" / "results", type=Path)
    ap.add_argument("--clean", action="store_true", help="remove ruled lines and crop tight first")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    with open(a.cells / "labels.csv") as f:
        rows = list(csv.DictReader(f))
    images = [Image.open(a.cells / r["file"]).convert("RGB") for r in rows]
    if a.clean:
        from clean_cell import clean_cell
        images = [clean_cell(im) for im in images]
    tag = "_clean" if a.clean else ""

    summary = []
    for key in a.models:
        proc = TrOCRProcessor.from_pretrained(MODELS[key])
        model = VisionEncoderDecoderModel.from_pretrained(MODELS[key]).eval()
        first, rest = digit_token_ids(proc.tokenizer)
        print(f"\n{key}: {len(first)} first / {len(rest)} continuation tokens allowed in restricted mode")

        for mode in ("free", "digits"):
            gen = dict(max_new_tokens=12, num_beams=4)
            if mode == "digits":
                gen["prefix_allowed_tokens_fn"] = lambda b, ids: first if len(ids) <= 1 else rest
            preds, t0 = [], time.time()
            for i in range(0, len(images), 16):
                pix = proc(images=images[i:i + 16], return_tensors="pt").pixel_values
                with torch.no_grad():
                    out = model.generate(pix, **gen)
                preds += proc.batch_decode(out, skip_special_tokens=True)
            secs = (time.time() - t0) / len(images)

            with open(a.out / f"{key}_{mode}{tag}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["file", "kind", "text", "pred_raw", "pred", "exact", "cer"])
                stats = {}
                for r, p in zip(rows, preds):
                    pn, ref = normalise(p), normalise(r["text"])
                    e, c = pn == ref, cer(pn, ref)
                    w.writerow([r["file"], r["kind"], r["text"], p, pn, int(e), f"{c:.3f}"])
                    for k in ("all", r["kind"], r["page"]):
                        s = stats.setdefault(k, [0, 0, 0.0])
                        s[0] += 1
                        s[1] += e
                        s[2] += c
            for k, (n, ex, c) in stats.items():
                summary.append((key, mode + tag, k, n, ex / n, c / n, secs))

    print(f"\n{'model':6} {'mode':7} {'subset':8} {'n':>4} {'exact':>7} {'CER':>6} {'s/cell':>7}")
    order = ["all", "integer", "decimal", "date", "time", "jul2010", "dec2016"]
    for key, mode, k, n, ex, c, secs in sorted(summary, key=lambda s: (s[0], s[1], order.index(s[2]))):
        print(f"{key:6} {mode:7} {k:8} {n:4d} {ex:7.1%} {c:6.3f} {secs:7.2f}")


if __name__ == "__main__":
    main()
