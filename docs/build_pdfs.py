"""Render docs/*.md to docs/pdf/*.pdf with WeasyPrint.

    uv run --no-project --with markdown --with weasyprint python docs/build_pdfs.py docs docs/pdf
"""
import re, sys
from pathlib import Path
import markdown
from weasyprint import HTML

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm;
  @bottom-left { content: "SJCET Weather Forecasting · %(title)s"; font: 8pt Inter; color: #777; }
  @bottom-right { content: "Page " counter(page) " of " counter(pages); font: 8pt Inter; color: #777; } }
body { font-family: Inter, "DejaVu Sans", sans-serif; font-size: 9.5pt; line-height: 1.45; color: #1d1d1f; }
h1 { font-size: 19pt; color: #0b3d5c; border-bottom: 2px solid #0b3d5c; padding-bottom: 4px; margin-top: 0; }
h2 { font-size: 13.5pt; color: #0b3d5c; margin-top: 18px; border-bottom: 1px solid #d0d7de; padding-bottom: 2px; page-break-after: avoid; }
h3 { font-size: 11pt; color: #164e63; margin-top: 12px; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8pt; background: #f2f4f7; padding: 0 2px; border-radius: 2px; }
pre { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 4px; padding: 7px 9px; white-space: pre-wrap; word-wrap: break-word; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 7.6pt; line-height: 1.3; }
table { border-collapse: collapse; width: 100%%; margin: 8px 0; font-size: 8.5pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th { background: #e8eef3; text-align: left; }
th, td { border: 1px solid #c9d1d9; padding: 3px 6px; vertical-align: top; }
blockquote { border-left: 3px solid #e3a008; background: #fffbeb; margin: 8px 0; padding: 4px 10px; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 14px 0; }
a { color: #0b62a4; text-decoration: none; }
"""

LIST = re.compile(r"^\s*([-*]|\d+\.) ")

def fix_lists(text):
    """Python-Markdown needs a blank line before a list (GitHub doesn't); add it outside code fences."""
    out, fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        prev = out[-1] if out else ""
        if not fence and LIST.match(line) and prev.strip() and not LIST.match(prev) and not prev.startswith((" ", "|")):
            out.append("")
        out.append(line)
    return "\n".join(out) + "\n"

def main(src_dir, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for md in sorted(Path(src_dir).glob("*.md")):
        text = fix_lists(md.read_text())
        title = re.search(r"^# (.+)$", text, re.M).group(1).split(":")[0].strip()
        body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
        body = re.sub(r'href="([\w-]+)\.md', r'href="\1.pdf', body)
        css = CSS % {"title": title.replace('"', "")}
        html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
        dst = out / (md.stem + ".pdf")
        HTML(string=html, base_url=str(md.parent)).write_pdf(dst)
        print(dst)

if __name__ == "__main__":
    main(*sys.argv[1:])
