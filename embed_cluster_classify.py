# embed_cluster_safe.py
# Ultra-safe version: skip problematic PDFs, no threading issues

import os, json, glob, argparse, pathlib, re, time, sys, random, unicodedata
from typing import List, Dict, Optional, Tuple
import signal

import numpy as np
import pandas as pd
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

# --- OpenAI ---
try:
    from openai import OpenAI
except:
    print("Install: pip install openai==1.*", file=sys.stderr)
    raise

# --- PDF/TXT parsing ---
from pypdf import PdfReader

# --- ML ---
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

try:
    import umap
    UMAP_AVAILABLE = True
except:
    UMAP_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Timeout handler (Unix-like systems)
# --------------------------
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# --------------------------
# Config
# --------------------------
ALLOWED_EXTS = {".txt", ".pdf"}
EMB_MODEL = "text-embedding-3-small"
CHARS_PER_CHUNK = 2000
BATCH_SIZE = 96
SLEEP = 0.25
MAX_RETRIES = 4
RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
sns.set_style("whitegrid")

# --------------------------
# Utils
# --------------------------
_surrogate_re = re.compile(r'[\ud800-\udfff]')

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = _surrogate_re.sub('', s)
    s = s.replace('\x00', ' ')
    s = ''.join(ch for ch in s if (ch >= ' ' or ch in '\n\t'))
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

def looks_like_pdf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(5).startswith(b"%PDF-")
    except:
        return False

def too_big(path: str, limit_mb: int) -> bool:
    if limit_mb <= 0:
        return False
    try:
        return os.path.getsize(path) / (1024*1024) > limit_mb
    except:
        return False

# --------------------------
# Safe PDF reader (NO threading, with manual timeout tracking)
# --------------------------
def read_pdf_safe(path: str, max_pages: int = 10) -> str:
    """
    Safe PDF reader that gives up quickly if parsing is slow.
    No threading/multiprocessing - just plain sequential with time checks.
    """
    if not looks_like_pdf(path):
        return ""
    
    try:
        reader = PdfReader(path, strict=False)
        n = len(reader.pages)
        limit = min(n, max_pages) if max_pages > 0 else n
        
        text = []
        start_time = time.time()
        
        for i in range(limit):
            # Check time before each page
            if time.time() - start_time > 8:  # 8 second hard limit
                return ""  # Give up
            
            try:
                page_text = reader.pages[i].extract_text() or ""
                text.append(page_text)
            except:
                continue
        
        return "\n".join(text).strip()
    except:
        return ""

def read_text_safe(path: str, max_pages: int = 10, size_limit_mb: int = 50) -> Tuple[str, Optional[str]]:
    """Ultra-safe text reader with no threading."""
    ext = pathlib.Path(path).suffix.lower()
    
    if ext == ".txt":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read(500000)  # Max 500KB text files
            return txt, None
        except:
            return "", "txt_read_failed"
    
    if ext == ".pdf":
        if too_big(path, size_limit_mb):
            return "", "too_large"
        
        start = time.time()
        txt = read_pdf_safe(path, max_pages)
        elapsed = time.time() - start
        
        if not txt:
            if elapsed > 5:
                return "", "timeout_or_failed"
            return "", "empty_or_unreadable"
        
        return txt, None
    
    return "", "unsupported_ext"

def chunk_text(text: str, size: int = CHARS_PER_CHUNK) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [text[i:i + size] for i in range(0, len(text), size)]

# --------------------------
# Embeddings
# --------------------------
def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in .env")
    return OpenAI(api_key=api_key)

def embed_batch(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            print(f"  API error (attempt {attempt}): {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(delay)
            delay *= 2.0

def embed_all(client: OpenAI, texts: List[str], model: str = EMB_MODEL) -> np.ndarray:
    vectors = []
    total = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding", total=total):
        batch = texts[i:i + BATCH_SIZE]
        vecs = embed_batch(client, batch, model)
        vectors.extend(vecs)
        time.sleep(SLEEP)
    
    return np.array(vectors, dtype=np.float32)

# --------------------------
# Clustering
# --------------------------
def auto_k(X: np.ndarray) -> int:
    best_k, best_s = 10, -1.0
    n = min(len(X), 8000)
    X_sample = X[np.random.choice(len(X), n, replace=False)]
    
    print("Finding optimal k...")
    for k in [5, 8, 10, 12, 15, 20]:
        if k >= len(X_sample):
            continue
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(X_sample)
        if len(set(labels)) > 1:
            s = silhouette_score(X_sample, labels)
            print(f"  k={k}: score={s:.3f}")
            if s > best_s:
                best_k, best_s = k, s
    
    return best_k

def find_medoids(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> Dict[int, int]:
    medoids = {}
    for c in range(centers.shape[0]):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        dist = np.linalg.norm(X[idx] - centers[c], axis=1)
        medoids[c] = idx[np.argmin(dist)]
    return medoids

# --------------------------
# Visualization
# --------------------------
def plot_all(df, out_dir, k):
    """Generate all plots."""
    
    # 1. UMAP
    if "umap_x" in df.columns:
        fig, ax = plt.subplots(figsize=(14, 10))
        scatter = ax.scatter(df["umap_x"], df["umap_y"], c=df["cluster"], 
                           cmap='tab20', alpha=0.5, s=50)
        rep = df[df["is_rep"]]
        ax.scatter(rep["umap_x"], rep["umap_y"], c=rep["cluster"],
                  cmap='tab20', s=400, marker='*', edgecolors='black', linewidths=2)
        
        for c in range(k):
            sub = df[df["cluster"] == c]
            ax.text(sub["umap_x"].mean(), sub["umap_y"].mean(), str(c),
                   fontsize=18, weight='bold', 
                   bbox=dict(boxstyle='circle', fc='white', ec='black', lw=2))
        
        ax.set_title(f"Document Clusters (k={k})\n⭐ = Representatives", fontsize=16, weight='bold')
        ax.set_xlabel("UMAP-1", fontsize=13)
        ax.set_ylabel("UMAP-2", fontsize=13)
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/cluster_umap.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 Saved: cluster_umap.png")
    
    # 2. Sizes
    fig, ax = plt.subplots(figsize=(12, 7))
    sizes = df.groupby("cluster").size().sort_index()
    colors = plt.cm.tab20(np.linspace(0, 1, len(sizes)))
    bars = ax.bar(range(k), sizes.values, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, sizes.values):
        ax.text(bar.get_x() + bar.get_width()/2, val, str(val),
               ha='center', va='bottom', fontsize=12, weight='bold')
    
    ax.set_title(f"Cluster Sizes (Total: {len(df)} chunks)", fontsize=16, weight='bold')
    ax.set_xlabel("Cluster ID", fontsize=13)
    ax.set_ylabel("Chunks", fontsize=13)
    ax.set_xticks(range(k))
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cluster_sizes.png", dpi=150)
    plt.close()
    print(f"  📊 Saved: cluster_sizes.png")
    
    # 3. Docs per cluster
    fig, ax = plt.subplots(figsize=(12, 7))
    doc_counts = df.groupby("cluster")["file"].nunique().sort_index()
    bars = ax.bar(range(k), doc_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, doc_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, val, str(val),
               ha='center', va='bottom', fontsize=12, weight='bold')
    
    ax.set_title(f"Documents per Cluster (Total: {df['file'].nunique()} files)", fontsize=16, weight='bold')
    ax.set_xlabel("Cluster ID", fontsize=13)
    ax.set_ylabel("Unique Files", fontsize=13)
    ax.set_xticks(range(k))
    plt.tight_layout()
    plt.savefig(f"{out_dir}/docs_per_cluster.png", dpi=150)
    plt.close()
    print(f"  📊 Saved: docs_per_cluster.png")

# --------------------------
# Main
# --------------------------
def main(data_dir: str, out_dir: str, k: Optional[int], max_pages: int, size_limit: int):
    os.makedirs(out_dir, exist_ok=True)
    client = get_client()

    # Gather
    files = []
    for ext in ALLOWED_EXTS:
        files.extend(glob.glob(f"{data_dir}/**/*{ext}", recursive=True))
    files = sorted(set(files))
    
    print(f"\n{'='*80}")
    print(f"📚 Found {len(files)} files")
    print(f"⚙️  Max pages: {max_pages}, Size limit: {size_limit}MB")
    print(f"{'='*80}\n")

    bad = []
    records = []
    
    # Parse WITHOUT threading (safest)
    print("Reading files (sequential, no threading)...\n")
    
    for i, fp in enumerate(files, 1):
        filename = pathlib.Path(fp).name
        print(f"[{i}/{len(files)}] {filename[:60]}", end=" ")
        sys.stdout.flush()  # Force print
        
        if too_big(fp, size_limit):
            print("⊗ TOO LARGE")
            bad.append((fp, "too_large"))
            continue
        
        t0 = time.time()
        txt, err = read_text_safe(fp, max_pages, size_limit)
        dt = time.time() - t0
        
        if err:
            print(f"✗ {err}")
            bad.append((fp, err))
            continue
        
        if not txt.strip():
            print("✗ empty")
            bad.append((fp, "empty"))
            continue
        
        txt = clean_text(txt)
        chunks = [clean_text(c) for c in chunk_text(txt) if clean_text(c)]
        
        if not chunks:
            print("✗ no chunks")
            bad.append((fp, "no_chunks"))
            continue
        
        for ci, ch in enumerate(chunks):
            records.append((fp, ci, ch))
        
        print(f"✓ {len(chunks)} chunks ({dt:.1f}s)")
    
    # Save bad files
    if bad:
        pd.DataFrame(bad, columns=["file", "reason"]).to_csv(f"{out_dir}/bad_files.csv", index=False)
        print(f"\n⚠️  Skipped {len(bad)} files → bad_files.csv")

    if not records:
        print("\n❌ No text extracted!")
        return

    # Build DataFrame
    print(f"\n✅ Extracted {len(records)} chunks from {len(set(r[0] for r in records))} files")
    
    df = pd.DataFrame(records, columns=["file", "chunk_id", "text"])
    df["id"] = [f"{p}::c{c}" for p, c in zip(df["file"], df["chunk_id"])]

    # Embedding cache
    cache_file = f"{out_dir}/embeddings.jsonl"
    cache = {}
    
    if os.path.exists(cache_file):
        print("Loading cache...")
        with open(cache_file, "r") as f:
            for line in f:
                try:
                    o = json.loads(line)
                    cache[o["id"]] = o["emb"]
                except:
                    continue
        print(f"  Loaded {len(cache)} cached")

    # Embed new
    to_embed = [i for i, _id in enumerate(df["id"]) if _id not in cache]
    
    if to_embed:
        print(f"\n🔮 Embedding {len(to_embed)} new chunks...")
        new_texts = df.iloc[to_embed]["text"].tolist()
        new_vecs = embed_all(client, new_texts)
        
        with open(cache_file, "a") as f:
            for idx, vec in zip(to_embed, new_vecs):
                _id = df.iloc[idx]["id"]
                f.write(json.dumps({"id": _id, "emb": vec.tolist()}) + "\n")
                cache[_id] = vec.tolist()

    # Build matrix
    df["emb"] = [np.array(cache[_id], dtype=np.float32) for _id in df["id"]]
    X = np.vstack(df["emb"].values)
    print(f"\n📐 Embedding matrix: {X.shape}")

    # PCA
    print("Running PCA...")
    n_comp = min(50, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    print(f"  Reduced to {X_pca.shape[1]}D")

    # KMeans
    if k is None:
        k = auto_k(X_pca)
        print(f"  Selected k={k}")
    
    print(f"Clustering with k={k}...")
    km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, batch_size=4096, n_init="auto")
    df["cluster"] = km.fit_predict(X_pca)

    # Medoids
    medoids = find_medoids(X_pca, df["cluster"].values, km.cluster_centers_)
    df["is_rep"] = False
    for c, idx in medoids.items():
        df.iloc[idx, df.columns.get_loc("is_rep")] = True
    
    print(f"  Found {len(medoids)} representatives")

    # UMAP
    if UMAP_AVAILABLE and len(df) >= 50:
        print("Computing UMAP...")
        reducer = umap.UMAP(random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
        umap_coords = reducer.fit_transform(X_pca)
        df["umap_x"] = umap_coords[:, 0]
        df["umap_y"] = umap_coords[:, 1]
        print("  ✓ UMAP done")

    # Save
    df["snippet"] = df["text"].str[:200].str.replace(r"\s+", " ", regex=True)
    
    cols = ["file", "chunk_id", "cluster", "is_rep", "snippet"]
    if "umap_x" in df.columns:
        cols += ["umap_x", "umap_y"]
    
    df.to_csv(f"{out_dir}/results.csv", index=False, columns=cols)
    print(f"\n📊 Results → results.csv")

    # Representatives
    rep = df[df["is_rep"]].copy()
    rep[["cluster", "file", "snippet"]].to_csv(f"{out_dir}/representatives.csv", index=False)
    print(f"⭐ Representatives → representatives.csv")

    # Visualizations
    print("\n🎨 Creating plots...")
    plot_all(df, out_dir, k)

    # Report (use ASCII-safe symbols)
    lines = [f"# Cluster Report (k={k})\n\n"]
    
    for c in range(k):
        sub = df[df["cluster"] == c]
        files_in_cluster = sub["file"].unique()
        
        lines.append(f"## Cluster {c}")
        lines.append(f"**Size**: {len(sub)} chunks, {len(files_in_cluster)} files\n")
        
        r = sub[sub["is_rep"]].head(1)
        if len(r):
            row = r.iloc[0]
            lines.append(f"### [STAR] Representative")
            lines.append(f"- File: `{pathlib.Path(row['file']).name}`")
            lines.append(f"- Text: {row.snippet}\n")
        
        lines.append("### Sample files:")
        for f in np.random.choice(files_in_cluster, min(3, len(files_in_cluster)), replace=False):
            lines.append(f"- `{pathlib.Path(f).name}`")
        lines.append("\n---\n")
    
    with open(f"{out_dir}/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📝 Report → report.md")

    # Summary
    print(f"\n{'='*80}")
    print("🎉 COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Processed: {df['file'].nunique()} files → {len(df)} chunks")
    print(f"⚠️  Skipped: {len(bad)} files")
    print(f"🎯 Clusters: {k}")
    print(f"⭐ Representatives: {len(rep)}")
    print(f"\n📁 Check: {out_dir}/")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--max_pages", type=int, default=10)
    ap.add_argument("--size_limit", type=int, default=50, help="Skip PDFs >N MB")
    args = ap.parse_args()

    print("\n🚀 SAFE MODE: No threading, aggressive timeouts\n")
    main(args.data_dir, args.out_dir, args.k, args.max_pages, args.size_limit)