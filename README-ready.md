# Ready-to-upload: MUVERA-like semantic news links aggregator

This package is pre-configured for **daily (06:00 Europe/Kyiv)** runs with MUVERA-like multi-vector rerank (BGE-M3).
Upload these files to your GitHub repo and run the workflow once.

## Files
- `aggregator_links.py` – main script (semantic rerank + categories + dedupe across runs)
- `requirements.txt` – deps (CPU)
- `.github/workflows/build.yml` – schedule (03:00 UTC ≈ 06:00 Kyiv)
- `muvera_config.json` – thresholds/limits without code edits
- `categories.json` – category rules (first-match wins)

## How to use
1) Upload everything into your repo (keep your existing `sources.txt`, `keywords.txt`, `stopwords.txt`, optional `blocked_domains.txt`).
2) Open **Actions** → **Build links (daily, MUVERA-like semantic rerank)** → **Run workflow** (first run).
3) Outputs per day:
   - `links.txt` – grouped by category
   - `links_by_category.json` – JSON
   - `archive_links.txt` – history
   - `seen.json` – already-seen links

## Tuning (optional)
- `muvera_config.json` – change `MAX_FETCH`, `THRESHOLD`, `TOPK_PER_CAT`.
- To disable semantic step temporarily: set `SEMANTIC_RERANK = False` atop `aggregator_links.py`.
