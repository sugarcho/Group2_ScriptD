# prefilter.py
# Filter your corpus before embedding/clustering by moving off-topic or broken files to a quarantine folder.

import os, re, sys, shutil, pathlib, argparse, json
from typing import Dict, List, Tuple

# --- deps ---
try:
    import yaml
except ImportError:
    print("Missing dependency: PyYAML. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# -----------------------
# Utilities
# -----------------------
def looks_like_pdf(path: str) -> bool:
    """Quick magic-header check for real PDFs."""
    try:
        with open(path, "rb") as f:
            return f.read(5).startswith(b"%PDF-")
    except Exception:
        return False

def read_pdf_head(pdf_path: str, pages: int = 2, max_chars: int = 2000) -> str:
    """Extract a small snippet from the first few pages of a PDF."""
    if PdfReader is None:
        return ""
    try:
        r = PdfReader(pdf_path, strict=False)
        text = []
        for i in range(min(pages, len(r.pages))):
            t = r.pages[i].extract_text() or ""
            text.append(t)
        s = " ".join(text).lower()
        s = re.sub(r"\s+", " ", s)
        return s[:max_chars]
    except Exception:
        return ""

def read_txt_head(txt_path: str, max_chars: int = 2000) -> str:
    """Read the first chunk of a TXT file."""
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return (f.read(max_chars) or "").lower()
    except Exception:
        return ""

def compile_rules(rules: Dict) -> Tuple[List[re.Pattern], List[re.Pattern]]:
    allow = rules.get("allow", []) or []
    deny  = rules.get("deny", []) or []
    allow_re = [re.compile(rf"\b{re.escape(k)}\b", re.I) for k in allow]
    deny_re  = [re.compile(rf"\b{re.escape(k)}\b", re.I) for k in deny]
    return allow_re, deny_re

# -----------------------
# Prefilter main
# -----------------------
def prefilter(
    data_dir: str,
    quarantine_dir: str,
    filters_yaml: str,
    pages: int = 2,
    max_chars: int = 2000,
    quarantine_fake_pdf: bool = True,
    quarantine_unreadable: bool = False
):
    if not os.path.exists(filters_yaml):
        raise FileNotFoundError(
            f"filters.yml not found at: {filters_yaml}\n"
            f"Create it (see example below) or pass an absolute path."
        )
    with open(filters_yaml, "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f) or {}
    allow_re, deny_re = compile_rules(rules)

    os.makedirs(quarantine_dir, exist_ok=True)

    n_keep = n_move = 0
    moved = []
    kept  = []

    for root, _, files in os.walk(data_dir):
        for fn in files:
            p = pathlib.Path(root) / fn
            ext = p.suffix.lower()
            if ext not in (".pdf", ".txt"):
                kept.append((str(p), "unsupported_extension")); n_keep += 1
                continue

            # Option A: quarantine fake PDFs (HTML renamed .pdf, etc.)
            if ext == ".pdf" and quarantine_fake_pdf and not looks_like_pdf(str(p)):
                dst = pathlib.Path(quarantine_dir) / fn
                try:
                    shutil.move(str(p), str(dst))
                    moved.append((str(p), "fake_pdf_header"))
                    n_move += 1
                except Exception as e:
                    kept.append((str(p), f"move_failed:{e}")); n_keep += 1
                continue

            # Peek at small portion
            if ext == ".pdf":
                head = read_pdf_head(str(p), pages=pages, max_chars=max_chars)
            else:
                head = read_txt_head(str(p), max_chars=max_chars)
            head_norm = re.sub(r"\s+", " ", head)

            # Option B: quarantine unreadables (no text)
            if quarantine_unreadable and not head_norm:
                dst = pathlib.Path(quarantine_dir) / fn
                try:
                    shutil.move(str(p), str(dst))
                    moved.append((str(p), "unreadable_head"))
                    n_move += 1
                except Exception as e:
                    kept.append((str(p), f"move_failed:{e}")); n_keep += 1
                continue

            # Allow / deny logic
            allow_hit = True if not allow_re else any(rx.search(head_norm) for rx in allow_re)
            deny_hit  = any(rx.search(head_norm) for rx in deny_re)

            if allow_hit and not deny_hit:
                kept.append((str(p), "kept"))
                n_keep += 1
            else:
                dst = pathlib.Path(quarantine_dir) / fn
                try:
                    shutil.move(str(p), str(dst))
                    moved.append((str(p), "deny_or_no_allow"))
                    n_move += 1
                except Exception as e:
                    kept.append((str(p), f"move_failed:{e}")); n_keep += 1

    # Summary
    summary = {
        "kept": n_keep,
        "moved_to_quarantine": n_move,
        "data_dir": data_dir,
        "quarantine": quarantine_dir,
        "flags": {
            "quarantine_fake_pdf": quarantine_fake_pdf,
            "quarantine_unreadable": quarantine_unreadable,
            "peek_pages": pages,
            "peek_max_chars": max_chars,
        }
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Optional: write logs
    kept_log = pathlib.Path(quarantine_dir) / "_prefilter_kept.tsv"
    moved_log = pathlib.Path(quarantine_dir) / "_prefilter_moved.tsv"
    try:
        with open(kept_log, "w", encoding="utf-8") as f:
            for path, reason in kept:
                f.write(f"{path}\t{reason}\n")
        with open(moved_log, "w", encoding="utf-8") as f:
            for path, reason in moved:
                f.write(f"{path}\t{reason}\n")
    except Exception:
        pass

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prefilter corpus by moving off-topic/bad files to quarantine.")
    ap.add_argument("data_dir", help="Root folder containing PDFs/TXTs.")
    ap.add_argument("quarantine_dir", help="Folder to move filtered files into.")
    ap.add_argument("filters_yaml", help="Path to filters.yml (with allow/deny lists).")
    ap.add_argument("--pages", type=int, default=2, help="How many PDF pages to peek at (default 2).")
    ap.add_argument("--max_chars", type=int, default=2000, help="Max chars to read from head (default 2000).")
    ap.add_argument("--no_quarantine_fake_pdf", action="store_true", help="Do NOT quarantine fake PDFs.")
    ap.add_argument("--quarantine_unreadable", action="store_true", help="Quarantine files with empty head text.")
    args = ap.parse_args()

    prefilter(
        data_dir=args.data_dir,
        quarantine_dir=args.quarantine_dir,
        filters_yaml=args.filters_yaml,
        pages=args.pages,
        max_chars=args.max_chars,
        quarantine_fake_pdf=not args.no_quarantine_fake_pdf,
        quarantine_unreadable=args.quarantine_unreadable,
    )
