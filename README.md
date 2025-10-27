# Group2_ScriptD


## Workflow Overview
Paper Download → Pre-filter → Text Extraction → Embedding → Clustering → Report

##  Data Retrieval Scripts
1. `uncertainty_d.py` - bioRxiv Paper Downloader
2. `scraper_crossref.py` - Crossref Open Access Papers
3. `scraper_semanticscholar.py` - Multi-Source Paper Retrieval

## Preprocessing & Filtering
`pre_filter.py` - Corpus Pre-filtering

Filtering rules:  
- Allow list: Files with specific keywords (e.g., "machine learning", "uncertainty")  
- Deny list: Files with excluded topics (e.g., "quantum", "astrophysics")  
- Fake PDF detection: Identifies HTML files disguised as PDFs  
- Unreadable detection: Files with no extractable text

## Embedding & Clustering
`embed_cluster_classify.py` - Core Analysis Pipeline
1. Text Extraction
    - Supports PDF and TXT files
    - Dual extraction with PyPDF and PDFMiner
    - Timeout protection (prevents hanging on large/corrupted PDFs)
    - Unicode cleaning and normalization

2. Text Chunking  

    ```python
   pythonCHARS_PER_CHUNK = 2000  # Characters per chunk
    ```
    - Splits long documents into manageable chunks
    - Maintains text coherence
    
3. Embedding
    - Uses OpenAI `text-embedding-3-small` model
    - Batch processing (96 texts/batch)
    - Auto-caching (avoids re-embedding)
    - Retry mechanism

4. Dimensionality Reduction & Clustering
    - PCA: Reduce to 50 dimensions
    - K-Means: Auto-select best k (via silhouette score)
    - UMAP: Optional 2D visualization

### Concurrency modes:
  ```bash
  # Default: Thread pool (Windows-safe)
  python embed_cluster_classify.py --data_dir ./papers --out_dir ./results
  
  # Single process (max compatibility)
  python embed_cluster_classify.py --data_dir ./papers --out_dir ./results --singleproc
  
  # Multi-process (fastest)
  python embed_cluster_classify.py --data_dir ./papers --out_dir ./results --multiproc
  ```
### Advanced parameters:
  ```bash
  --timeout 25           # PDF timeout (seconds)
  --max_pages 20         # Max pages per PDF
  --slow_threshold 10    # Slow file threshold (seconds)
  --size_skip_mb 50      # Skip files larger than this (MB)
  --k 15                 # Fixed cluster count (otherwise auto)
  ```

### Output files:
  - `clusters_and_embeddings.csv`: Main results table
  - `cluster_report.md`: Human-readable cluster summary
  - `embeddings.jsonl`: Embedding cache
  - `bad_files.csv`: Failed files log
  - `slow_files.csv`: Slow files log

### Configuration Files
  - `filters.yml` - Topic Filtering Rules
  - `requirements.txt` - Python Dependencies

## Usage Guide
1. Environment Setup
    - write infor into `.env` file
2. Download Papers
    ``` bash
    # bioRxiv
    python uncertainty_d.py
    
    # Crossref
    python scraper_crossref.py
    
    # Multi-source (recommended)
    python scraper_semanticscholar.py
    ```
3. Pre-filter
    ``` bash
    python pre_filter.py \
      papers_uncertainty_prediction_ml \
      quarantine \
      filters.yml \
      --quarantine_unreadable
    ```
4. Embed and Cluster
    ``` bash
    python embed_cluster_classify.py \
      --data_dir papers_uncertainty_prediction_ml \
      --out_dir analysis_results \
      --timeout 30 \
      --max_pages 20
    ```
5. Analyze Results
    - `cluster_report.md`: Summary of each cluster with representative docs
    - `clusters_and_embeddings.csv`: Full data table for further analysis
    - If UMAP: Visualize umap_x, umap_y coordinates