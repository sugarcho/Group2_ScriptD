# pre_filter_v2.py
# Enhanced version with smart pairing detection

import os
import re
import sys
import shutil
import pathlib
import argparse
import json
from typing import Dict, List, Tuple, Set
from collections import defaultdict

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
# Pairing detection
# -----------------------
def find_paired_file(filepath: pathlib.Path, all_files: Set[str]) -> str:
    """
    Find the paired file for a given PDF or TXT.
    Handles both simple pairs (paper.pdf + paper.txt) and 
    metadata pairs (paper.pdf + paper_metadata.txt).
    """
    stem = filepath.stem
    ext = filepath.suffix.lower()
    parent = filepath.parent
    
    if ext == '.pdf':
        # Look for .txt counterparts
        candidates = [
            parent / f"{stem}.txt",           # exact match
            parent / f"{stem}_metadata.txt",  # metadata variant
        ]
    elif ext == '.txt':
        # Look for .pdf counterparts
        if stem.endswith('_metadata'):
            base_stem = stem[:-9]  # remove "_metadata"
            candidates = [
                parent / f"{base_stem}.pdf",
            ]
        else:
            candidates = [
                parent / f"{stem}.pdf",
            ]
    else:
        return None
    
    # Return first matching candidate
    for candidate in candidates:
        if str(candidate) in all_files:
            return str(candidate)
    
    return None

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
    """Compile allow/deny keyword lists into regex patterns."""
    allow = rules.get("allow", []) or []
    deny  = rules.get("deny", []) or []
    
    allow_re = [re.compile(rf"\b{re.escape(k)}\b", re.I) for k in allow]
    deny_re  = [re.compile(rf"\b{re.escape(k)}\b", re.I) for k in deny]
    
    return allow_re, deny_re

# -----------------------
# Prefilter main (v2)
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
    """
    Enhanced prefilter with smart pairing detection.
    Handles both exact pairs (paper.pdf + paper.txt) and 
    metadata pairs (paper.pdf + paper_metadata.txt).
    """
    if not os.path.exists(filters_yaml):
        raise FileNotFoundError(f"filters.yml not found at: {filters_yaml}")
    
    with open(filters_yaml, "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f) or {}
    
    allow_re, deny_re = compile_rules(rules)
    os.makedirs(quarantine_dir, exist_ok=True)

    # Phase 1: Collect all files
    all_files = set()
    for root, _, files in os.walk(data_dir):
        for fn in files:
            p = pathlib.Path(root) / fn
            if p.suffix.lower() in ('.pdf', '.txt'):
                all_files.add(str(p))
    
    # Phase 2: Evaluate each file and mark for quarantine
    files_to_quarantine = set()  # Full file paths
    file_reasons = {}  # filepath -> reason
    
    for filepath_str in all_files:
        p = pathlib.Path(filepath_str)
        ext = p.suffix.lower()
        
        # Check 1: Fake PDF detection
        if ext == ".pdf" and quarantine_fake_pdf:
            if not looks_like_pdf(filepath_str):
                files_to_quarantine.add(filepath_str)
                file_reasons[filepath_str] = 'fake_pdf_header'
                continue
        
        # Extract text snippet
        if ext == ".pdf":
            head = read_pdf_head(filepath_str, pages=pages, max_chars=max_chars)
        else:
            head = read_txt_head(filepath_str, max_chars=max_chars)
        
        head_norm = re.sub(r"\s+", " ", head)
        
        # Check 2: Unreadable files
        if quarantine_unreadable and not head_norm:
            files_to_quarantine.add(filepath_str)
            file_reasons[filepath_str] = 'unreadable'
            continue
        
        # Check 3: Allow/Deny keyword filtering
        allow_hit = True if not allow_re else any(rx.search(head_norm) for rx in allow_re)
        deny_hit  = any(rx.search(head_norm) for rx in deny_re)
        
        if not allow_hit or deny_hit:
            reason = 'deny_keyword' if deny_hit else 'no_allow_keyword'
            files_to_quarantine.add(filepath_str)
            file_reasons[filepath_str] = reason
    
    # Phase 3: Add paired files to quarantine set
    files_to_quarantine_final = set(files_to_quarantine)
    
    for filepath_str in files_to_quarantine:
        p = pathlib.Path(filepath_str)
        paired_file = find_paired_file(p, all_files)
        
        if paired_file and paired_file not in files_to_quarantine:
            files_to_quarantine_final.add(paired_file)
            file_reasons[paired_file] = 'paired_with_quarantined'
    
    # Phase 4: Move files
    n_keep = 0
    n_move = 0
    moved = []
    kept = []
    
    for filepath_str in all_files:
        p = pathlib.Path(filepath_str)
        dst_path = pathlib.Path(quarantine_dir) / p.name
        
        if filepath_str in files_to_quarantine_final:
            # Move to quarantine
            try:
                shutil.move(filepath_str, str(dst_path))
                reason = file_reasons.get(filepath_str, 'unknown')
                moved.append((filepath_str, reason))
                n_move += 1
            except Exception as e:
                kept.append((filepath_str, f"move_failed:{e}"))
                n_keep += 1
        else:
            # Keep in original location
            kept.append((filepath_str, 'kept'))
            n_keep += 1
    
    # Summary
    summary = {
        "kept": n_keep,
        "moved_to_quarantine": n_move,
        "data_dir": data_dir,
        "quarantine": quarantine_dir,
        "pairing_strategy": "smart (handles exact + metadata pairs)",
        "flags": {
            "quarantine_fake_pdf": quarantine_fake_pdf,
            "quarantine_unreadable": quarantine_unreadable,
            "peek_pages": pages,
            "peek_max_chars": max_chars,
        }
    }
    
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    
    # Write logs
    kept_log = pathlib.Path(quarantine_dir) / "_prefilter_kept.tsv"
    moved_log = pathlib.Path(quarantine_dir) / "_prefilter_moved.tsv"
    
    try:
        with open(kept_log, "w", encoding="utf-8") as f:
            f.write("filepath\treason\n")
            for path, reason in sorted(kept):
                f.write(f"{path}\t{reason}\n")
        
        with open(moved_log, "w", encoding="utf-8") as f:
            f.write("filepath\treason\n")
            for path, reason in sorted(moved):
                f.write(f"{path}\t{reason}\n")
        
        print(f"\n✅ Logs written to:")
        print(f"   - {kept_log}")
        print(f"   - {moved_log}")
    except Exception as e:
        print(f"⚠️  Failed to write logs: {e}", file=sys.stderr)

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Prefilter corpus with smart PDF/TXT pairing (handles metadata files too)."
    )
    ap.add_argument("data_dir", help="Root folder containing PDFs/TXTs.")
    ap.add_argument("quarantine_dir", help="Folder to move filtered files into.")
    ap.add_argument("filters_yaml", help="Path to filters.yml (with allow/deny lists).")
    ap.add_argument("--pages", type=int, default=3, help="How many PDF pages to peek at (default 3).")
    ap.add_argument("--max_chars", type=int, default=3000, help="Max chars to read from head (default 3000).")
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