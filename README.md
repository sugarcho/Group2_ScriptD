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

Usage:  
```bash
python pre_filter.py
```  
    
Options:  
``` bash
--pages 2                     # PDF pages to peek at (default: 2)  
--max_chars 2000              # Max characters to read (default: 2000)  
--no_quarantine_fake_pdf      # Disable fake PDF detection  
--quarantine_unreadable       # Enable unreadable file detection  
```
    
## Embedding & Clustering
`embed_cluster_classify.py` - Core Analysis Pipeline
1. Text Extraction
    - Supports PDF and TXT files
    - Sequential processing with 8-second timeout per PDF page
    - Size limit: Skips files larger than threshold (default: 50MB)
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
    - Auto-caching: Saves to `embeddings.jsonl` (enables incremental processing)
    - Retry mechanism: Up to 4 retries with exponential backoff

4. Dimensionality Reduction & Clustering
    - PCA: Reduce to 50 dimensions
    - K-Means: Auto-select best k via silhouette score (tests k ∈ {5, 8, 10, 12, 15, 20})
    - UMAP: Optional 2D visualization

### Concurrency modes:
  ```bash
    python embed_cluster_classify.py --data_dir ./papers --out_dir ./results
  ```
### Advanced parameters:
  ```bash
    --data_dir ./papers       # Required: Input folder
    --out_dir ./results       # Required: Output folder
    --k 15                    # Fixed cluster count (otherwise auto)
    --max_pages 10           # Max pages per PDF (default: 10)
    --size_limit 50          # Skip PDFs larger than N MB (default: 50)
  ```

### Output files:
  - `results.csv`: Main results table with clusters, snippets, UMAP coordinates
  - `representatives.csv`: Representative text from each cluster (medoids)
  - `embeddings.jsonl`: Embedding cache for incremental processing
  - `bad_files.csv`: Failed files with error reasons
  - `report.md`: Human-readable cluster summary
  - `cluster_sizes.png`: Bar chart of chunks per cluster
  - `docs_per_cluster.png`: Bar chart of documents per cluster
  - `cluster_umap.png`: 2D UMAP visualization with representatives

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
      ./raw_papers \
      ./quarantine \
      ./filters.yaml \
      --quarantine_unreadable
    ```
4. Embed and Cluster
    ``` bash
    python embed_cluster_classify.py \
      --data_dir \
      ./raw_papers \
      --out_dir \
      ./results \
      --max_pages 15    
    ```
5. Analyze Results
    - `cluster_report.md`: Summary of each cluster with representative docs
    - `clusters_and_embeddings.csv`: Full data table for further analysis
    - If UMAP: Visualize umap_x, umap_y coordinates
