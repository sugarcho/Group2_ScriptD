# embed_cluster_classify.py
# End-to-end pipeline with pool-mode switches and CLI knobs for timeout/pages/slow/size.
# Default: thread pool (safe on Windows). Use --singleproc or --multiproc if desired.

import os, json, glob, argparse, pathlib, re, time, sys, random, logging, warnings, unicodedata
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Load env (.env) ---
from dotenv import load_dotenv
load_dotenv()

# --- Quiet noisy logs ---
warnings.filterwarnings("ignore", category=UserWarning)
for name in ["pypdf", "pdfminer", "pdfminer.pdfinterp", "pdfminer.layout"]:
    logging.getLogger(name).setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# --- OpenAI ---
try:
    from openai import OpenAI
except Exception:
    print("Install OpenAI SDK v1:  pip install openai==1.*", file=sys.stderr)
    raise

# --- PDF/TXT parsing ---
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text

# --- ML ---
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# optional UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# concurrency
from concurrent.futures import TimeoutError as FuturesTimeoutError, ThreadPoolExecutor, ProcessPoolExecutor

# --------------------------
# Defaults (can be overridden by CLI)
# --------------------------
ALLOWED_EXTS = {".txt", ".pdf"}
EMB_MODEL = "text-embedding-3-small"
CHARS_PER_CHUNK = 2000
BATCH_SIZE = 96
SLEEP_BETWEEN_CALLS = 0.25
MAX_RETRIES = 4

# Will be overridden by CLI:
PDF_MAX_PAGES = 20
PDF_TIMEOUT_S = 25
SLOW_FILE_THRESHOLD_S = 10
SIZE_SKIP_MB = 0  # 0 = off

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# --------------------------
# OpenAI client
# --------------------------
def _ensure_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY missing. Create .env with:\nOPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx\n"
        )
    return OpenAI(api_key=api_key)

# --------------------------
# Unicode sanitizer
# --------------------------
_surrogate_re = re.compile(r'[\ud800-\udfff]')

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = _surrogate_re.sub('', s)                # drop lone surrogates
    s = s.replace('\x00', ' ')                  # drop NULs
    s = ''.join(ch for ch in s if (ch >= ' ' or ch in '\n\t'))  # keep printable
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.encode('utf-8', 'ignore').decode('utf-8', 'ignore')   # final guard
    return s

# --------------------------
# File helpers
# --------------------------
def _looks_like_pdf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(5)
        return head.startswith(b"%PDF-")
    except Exception:
        return False

def _too_big(path: str) -> bool:
    if SIZE_SKIP_MB <= 0:
        return False
    try:
        return os.path.getsize(path) > SIZE_SKIP_MB * 1024 * 1024
    except Exception:
        return False

def _read_pdf_with_pypdf(path: str, max_pages: int) -> str:
    reader = PdfReader(path, strict=False)
    text = []
    n = len(reader.pages)
    limit = min(n, max_pages) if max_pages and max_pages > 0 else n
    for i in range(limit):
        page = reader.pages[i]
        t = page.extract_text() or ""
        text.append(t)
    return "\n".join(text).strip()

def _read_pdf_with_pdfminer(path: str, max_pages: int) -> str:
    kwargs = {}
    if max_pages and max_pages > 0:
        kwargs["maxpages"] = max_pages
    return (pdfminer_extract_text(path, **kwargs) or "").strip()

def _read_pdf_worker(path: str, max_pages: int) -> str:
    if not _looks_like_pdf(path):
        return ""
    try:
        txt = _read_pdf_with_pypdf(path, max_pages=max_pages)
        if len(txt) >= 50:
            return txt
    except Exception:
        pass
    try:
        return _read_pdf_with_pdfminer(path, max_pages=max_pages)
    except Exception:
        return ""

def read_text_singleproc(path: str) -> Tuple[str, Optional[str]]:
    """Single-process fallback with wall-clock cutoff (no futures)."""
    ext = pathlib.Path(path).suffix.lower()
    t0 = time.time()

    if ext == ".txt":
        for enc in ("utf-8", "latin-1"):
            try:
                with open(path, "r", encoding=enc, errors="ignore") as f:
                    txt = f.read()
                return txt, None
            except Exception:
                continue
        return "", "txt_read_failed"

    if ext == ".pdf":
        if not _looks_like_pdf(path):
            return "", "not_real_pdf"
        if _too_big(path):
            return "", "skipped_large_file"
        try:
            txt = _read_pdf_worker(path, PDF_MAX_PAGES)
        except Exception:
            return "", "pdf_parse_failed"
        dt = time.time() - t0
        if dt > PDF_TIMEOUT_S:
            return "", "timeout"
        return (txt, None) if txt.strip() else ("", "empty_or_unreadable")

    return "", "unsupported_extension"

def read_text_via_pool(path: str, pool, use_timeout: bool = True) -> Tuple[str, Optional[str]]:
    ext = pathlib.Path(path).suffix.lower()

    if ext == ".txt":
        for enc in ("utf-8", "latin-1"):
            try:
                with open(path, "r", encoding=enc, errors="ignore") as f:
                    txt = f.read()
                return txt, None
            except Exception:
                continue
        return "", "txt_read_failed"

    if ext == ".pdf":
        if not _looks_like_pdf(path):
            return "", "not_real_pdf"
        if _too_big(path):
            return "", "skipped_large_file"
        fut = pool.submit(_read_pdf_worker, path, PDF_MAX_PAGES)
        try:
            txt = fut.result(timeout=PDF_TIMEOUT_S if use_timeout else None)
            return (txt, None) if txt.strip() else ("", "empty_or_unreadable")
        except FuturesTimeoutError:
            try: fut.cancel()
            except Exception: pass
            return "", "timeout"
        except Exception:
            return "", "pdf_parse_failed"

    return "", "unsupported_extension"

def chunk_text(text: str, chars_per_chunk: int = CHARS_PER_CHUNK) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [text[i:i + chars_per_chunk] for i in range(0, len(text), chars_per_chunk)]

# --------------------------
# Embeddings (cached) with retries
# --------------------------
def _embed_batch(client: OpenAI, inputs: List[str], model: str) -> List[List[float]]:
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.embeddings.create(model=model, input=inputs)
            return [d.embedding for d in resp.data]
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(delay)
            delay *= 2.0

def embed_texts(client: OpenAI, texts: List[str], model: str = EMB_MODEL) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        vecs = _embed_batch(client, batch, model=model)
        vectors.extend(vecs)
        time.sleep(SLEEP_BETWEEN_CALLS)
    return np.array(vectors, dtype=np.float32)

# --------------------------
# Clustering helpers
# --------------------------
def find_k_by_silhouette(X: np.ndarray, k_candidates: List[int]) -> int:
    best_k, best_score = None, -1.0
    n = len(X)
    X_eval = X if n <= 12000 else X[np.random.choice(n, size=10000, replace=False)]
    for k in k_candidates:
        if k <= 1 or k >= len(X_eval):
            continue
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, batch_size=2048, n_init="auto")
        labels = km.fit_predict(X_eval)
        if len(set(labels)) <= 1:
            continue
        score = silhouette_score(X_eval, labels, metric="euclidean")
        if score > best_score:
            best_k, best_score = k, score
    return best_k or max(2, min(30, len(X)//200))

def nearest_medoid_indices(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> Dict[int, int]:
    medoid_idx = {}
    for c in range(centroids.shape[0]):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        d = np.linalg.norm(X[members] - centroids[c], axis=1)
        m = members[int(np.argmin(d))]
        medoid_idx[c] = m
    return medoid_idx

# --------------------------
# Optional supervised labels
# --------------------------
def load_labels_csv(path: str) -> Dict[str, str]:
    df = pd.read_csv(path)
    df["file_path"] = df["file_path"].astype(str)
    df["label"] = df["label"].astype(str)
    return dict(zip(df["file_path"], df["label"]))

# --------------------------
# Main
# --------------------------
def main(
    data_dir: str,
    out_dir: str,
    labels_csv: Optional[str] = None,
    k: Optional[int] = None,
    min_docs_for_umap: int = 50,
    singleproc: bool = False,
    multiproc: bool = False,
    timeout: int = 25,
    max_pages: int = 20,
    slow_threshold: int = 10,
    size_skip_mb: int = 0
):
    global PDF_TIMEOUT_S, PDF_MAX_PAGES, SLOW_FILE_THRESHOLD_S, SIZE_SKIP_MB
    PDF_TIMEOUT_S = int(timeout)
    PDF_MAX_PAGES = int(max_pages)
    SLOW_FILE_THRESHOLD_S = int(slow_threshold)
    SIZE_SKIP_MB = int(size_skip_mb)

    os.makedirs(out_dir, exist_ok=True)
    client = _ensure_openai_client()

    # Gather files
    files = []
    for ext in ALLOWED_EXTS:
        files.extend(glob.glob(os.path.join(data_dir, f"**/*{ext}"), recursive=True))
    files = sorted(set(files))
    print(f"Reading & chunking {len(files)} files...")

    bad: List[Tuple[str, str]] = []
    slow: List[Tuple[str, float]] = []
    records: List[Tuple[str, int, str]] = []

    # Choose mode
    mode = "threads"
    if singleproc:
        mode = "single"
    elif multiproc:
        mode = "processes"

    try:
        if mode == "single":
            for fp in tqdm(files):
                if _too_big(fp):
                    bad.append((fp, "skipped_large_file")); continue
                t0 = time.time()
                txt, err = read_text_singleproc(fp)
                dt = time.time() - t0
                if dt > SLOW_FILE_THRESHOLD_S: slow.append((fp, dt))
                if err or not txt.strip():
                    bad.append((fp, err or "empty_or_unreadable")); continue
                txt = clean_text(txt)
                if not txt:
                    bad.append((fp, "cleaned_to_empty")); continue
                for ci, ch in enumerate([clean_text(c) for c in chunk_text(txt)]):
                    if ch: records.append((fp, ci, ch))

        elif mode == "processes":
            with ProcessPoolExecutor(max_workers=2) as pool:
                for fp in tqdm(files):
                    if _too_big(fp):
                        bad.append((fp, "skipped_large_file")); continue
                    t0 = time.time()
                    txt, err = read_text_via_pool(fp, pool, use_timeout=True)
                    dt = time.time() - t0
                    if dt > SLOW_FILE_THRESHOLD_S: slow.append((fp, dt))
                    if err or not txt.strip():
                        bad.append((fp, err or "empty_or_unreadable")); continue
                    txt = clean_text(txt)
                    if not txt:
                        bad.append((fp, "cleaned_to_empty")); continue
                    for ci, ch in enumerate([clean_text(c) for c in chunk_text(txt)]):
                        if ch: records.append((fp, ci, ch))
        else:  # threads (default)
            with ThreadPoolExecutor(max_workers=2) as pool:
                for fp in tqdm(files):
                    if _too_big(fp):
                        bad.append((fp, "skipped_large_file")); continue
                    t0 = time.time()
                    txt, err = read_text_via_pool(fp, pool, use_timeout=True)
                    dt = time.time() - t0
                    if dt > SLOW_FILE_THRESHOLD_S: slow.append((fp, dt))
                    if err or not txt.strip():
                        bad.append((fp, err or "empty_or_unreadable")); continue
                    txt = clean_text(txt)
                    if not txt:
                        bad.append((fp, "cleaned_to_empty")); continue
                    for ci, ch in enumerate([clean_text(c) for c in chunk_text(txt)]):
                        if ch: records.append((fp, ci, ch))
    except KeyboardInterrupt:
        print("Interrupted; writing logs…")

    # Logs
    if bad:
        pd.DataFrame(bad, columns=["file_path", "reason"]).to_csv(os.path.join(out_dir, "bad_files.csv"), index=False)
        print(f"Skipped {len(bad)} files; details -> {os.path.join(out_dir, 'bad_files.csv')}")
    if slow:
        pd.DataFrame(slow, columns=["file_path", "seconds"]).to_csv(os.path.join(out_dir, "slow_files.csv"), index=False)
        print(f"Logged {len(slow)} slow files; details -> {os.path.join(out_dir, 'slow_files.csv')}")

    if not records:
        print("No usable text found. Exiting.")
        return

    df = pd.DataFrame(records, columns=["file_path", "chunk_id", "text"])
    df["id"] = [f"{p}::chunk{c}" for p, c in zip(df["file_path"], df["chunk_id"])]
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # Embedding cache
    cache_path = os.path.join(out_dir, "embeddings.jsonl")
    cache: Dict[str, List[float]] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line); cache[o["id"]] = o["embedding"]
                except Exception:
                    continue

    ids = df["id"].tolist()
    to_embed_idx = [i for i, _id in enumerate(ids) if _id not in cache]
    print(f"Embedding {len(to_embed_idx)} new chunks (cached: {len(cache)})...")

    if to_embed_idx:
        new_texts = df.iloc[to_embed_idx]["text"].tolist()
        new_vecs = embed_texts(client, new_texts, model=EMB_MODEL)
        with open(cache_path, "a", encoding="utf-8") as f:
            for idx, vec in zip(to_embed_idx, new_vecs):
                rec_id = df.iloc[idx]["id"]
                f.write(json.dumps({"id": rec_id, "embedding": vec.tolist()}) + "\n")
                cache[rec_id] = vec.tolist()

    # Build matrix
    df["embedding"] = [np.array(cache[_id], dtype=np.float32) for _id in df["id"]]
    X = np.vstack(df["embedding"].values)

    # Optional supervised
    df["true_label"] = None
    df["pred_label"] = None
    if labels_csv and os.path.exists(labels_csv):
        print("[Supervised] Training from labels.csv…")
        file_to_label = load_labels_csv(labels_csv)
        df["true_label"] = df["file_path"].map(file_to_label)
        have_labels = df.dropna(subset=["true_label"]).copy()
        if len(have_labels) > 0:
            files_l = sorted(have_labels["file_path"].unique().tolist())
            cutoff = int(0.8 * len(files_l))
            train_files, test_files = set(files_l[:cutoff]), set(files_l[cutoff:])
            X_train = np.vstack(have_labels[have_labels["file_path"].isin(train_files)]["embedding"].values)
            y_train = have_labels[have_labels["file_path"].isin(train_files)]["true_label"].values
            X_test  = np.vstack(have_labels[have_labels["file_path"].isin(test_files)]["embedding"].values)
            y_test  = have_labels[have_labels["file_path"].isin(test_files)]["true_label"].values
            test_rows = have_labels[have_labels["file_path"].isin(test_files)].index

            clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000))
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = (y_pred == y_test).mean()
            print(f"[Supervised] Test accuracy (by chunk): {acc:.3f}")
            df.loc[test_rows, "pred_label"] = y_pred

    # PCA → KMeans
    print("PCA → KMeans…")
    pca = PCA(n_components=min(50, X.shape[1]), random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    if k is None:
        k = find_k_by_silhouette(X_pca, k_candidates=[5, 8, 10, 12, 15, 20, 25, 30])
        print(f"Auto-selected k={k}")

    kmeans = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, batch_size=4096, n_init="auto")
    labels = kmeans.fit_predict(X_pca)
    df["cluster"] = labels

    # Representatives
    medoid_idx = nearest_medoid_indices(X_pca, labels, kmeans.cluster_centers_)
    df["is_representative"] = False
    for c, idx in medoid_idx.items():
        df.iloc[idx, df.columns.get_loc("is_representative")] = True

    # UMAP
    if UMAP_AVAILABLE and len(df) >= min_docs_for_umap:
        print("Computing UMAP 2D layout…")
        reducer = umap.UMAP(random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1, metric="euclidean")
        emb2d = reducer.fit_transform(X_pca)
        df["umap_x"] = emb2d[:, 0]
        df["umap_y"] = emb2d[:, 1]

    # Outputs
    df["snippet"] = df["text"].str.slice(0, 240).str.replace(r"\s+", " ", regex=True)
    out_csv = os.path.join(out_dir, "clusters_and_embeddings.csv")
    cols = ["file_path","chunk_id","cluster","is_representative","true_label","pred_label","snippet"]
    if "umap_x" in df.columns:
        cols += ["umap_x","umap_y"]
    df.to_csv(out_csv, index=False, columns=cols)
    print(f"Wrote: {out_csv}")

    report_lines = [f"# Cluster Report (k={k})\n"]
    for c in range(k):
        sub = df[df["cluster"] == c]
        report_lines.append(f"## Cluster {c}  (n={len(sub)})")
        rep = sub[sub["is_representative"]].head(1)
        if len(rep):
            r = rep.iloc[0]
            report_lines.append(f"- **Representative file**: `{r.file_path}` (chunk {r.chunk_id})")
            report_lines.append(f"- **Snippet**: {r.snippet}\n")
        if len(sub) > 0:
            examples = sub.sample(min(3, len(sub)), random_state=RANDOM_STATE)
            for _, row in examples.iterrows():
                report_lines.append(f"  - `{row.file_path}` (chunk {row.chunk_id}): {row.snippet}")
        report_lines.append("")
    out_md = os.path.join(out_dir, "cluster_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Wrote: {out_md}")

    print("\nTips:")
    print(f"- Mode: {mode}. Default threads are stable on Windows. Use --singleproc for max compatibility; --multiproc for speed.")
    print(f"- Timeout {PDF_TIMEOUT_S}s, max_pages {PDF_MAX_PAGES}, slow_threshold {SLOW_FILE_THRESHOLD_S}s, size_skip_mb {SIZE_SKIP_MB}MB.")
    print("- Check bad_files.csv / slow_files.csv for problematic PDFs. Cache avoids re-embedding unchanged chunks.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder containing your PDFs/TXTs (recursively).")
    ap.add_argument("--out_dir", required=True, help="Output folder for CSV/MD and cache.")
    ap.add_argument("--labels_csv", default=None, help="Optional CSV with columns: file_path,label (for supervised).")
    ap.add_argument("--k", type=int, default=None, help="Optional fixed number of clusters. If omitted, auto-selected.")
    ap.add_argument("--singleproc", action="store_true", help="Parse in a single process (no pool).")
    ap.add_argument("--multiproc", action="store_true", help="Use a ProcessPoolExecutor (fast, can be flaky on Windows).")
    ap.add_argument("--timeout", type=int, default=25, help="Per-PDF timeout seconds (default 25).")
    ap.add_argument("--max_pages", type=int, default=20, help="Max pages per PDF to parse (default 20; 0 = all).")
    ap.add_argument("--slow_threshold", type=int, default=10, help="Log files slower than this many seconds (default 10).")
    ap.add_argument("--size_skip_mb", type=int, default=0, help="Skip PDFs larger than this many MB (default 0=off).")
    args = ap.parse_args()

    main(
        args.data_dir,
        args.out_dir,
        args.labels_csv,
        args.k,
        singleproc=args.singleproc,
        multiproc=args.multiproc,
        timeout=args.timeout,
        max_pages=args.max_pages,
        slow_threshold=args.slow_threshold,
        size_skip_mb=args.size_skip_mb,
    )


