# News & OSINT Links Aggregator
_Async Multi-Provider · MUVERA-style semantic rerank · SQLite Vector Cache · strict freshness (≤24h)_

This tool collects **fresh, relevant links** across telecom, caller ID/robocalls, parental control, OSINT, cybersecurity, and related niches. 
It no longer relies solely on RSS: it uses an **Async Multi-Provider architecture** to pull data directly from **ArXiv** (research papers), **Crossref**, **Hacker News**, **GitHub**, and your custom RSS feeds. 

It filters by **keywords/stopwords**, enforces **≤ 24 hours** freshness, **reranks semantically** (MUVERA-like hybrid with `BAAI/bge-m3`), and routes items into **categories**. All state, logs, and heavy neural network embeddings are cached in a lightning-fast **SQLite database** to bypass GitHub Actions limits.

---

## Features

- **Multi-Provider Architecture:** Simultaneously fetches data from RSS/Atom, GitHub Trending APIs, Hacker News Firebase, ArXiv, and Crossref.
- **Blazing Fast Async I/O:** Network requests are powered by `aiohttp` and `asyncio`, fetching hundreds of articles in seconds.
- **SQLite Database & Vector Cache:** `seen.json` and `reasons.jsonl` are replaced by `heylocate.db`. Text contents and BGE-M3 Dense Vectors (pickled BLOBs) are cached in DB, saving massive CPU cycles on re-runs.
- **Freshness gate:** only items **published within the last 24 hours** (`MAX_AGE_DAYS=1`, `REQUIRE_PUBDATE=true`).
- **Keyword prefilter + stopword filter** before heavy scoring (includes `NoneType` safety for missing API descriptions).
- **MUVERA-style semantic rerank (optional):** hybrid dense + sparse + ColBERT using `BAAI/bge-m3`.
- **Categorization:** rule-based routing via `categories.json` (`include_any`, `include_all`, `exclude_any`, optional `domains`).
- **GitHub Actions ready:** CPU Torch wheels, auto-migrations, daily schedule, and GITHUB_TOKEN integration for API limits.

---

## Repository Layout

.
├─ aggregator_links.py          # main async script
├─ muvera_config.json           # engine configuration (freshness, thresholds)
├─ categories.json              # topic routing rules
├─ sources.txt                  # RSS/Atom feeds (one URL per line)
├─ keywords.txt                 # positive terms (one per line)
├─ stopwords.txt                # optional global negative terms
├─ blocked_domains.txt          # optional domain blocklist (can be empty)
│
├─ links.txt                    # OUTPUT: human-readable, grouped by category
├─ links_by_category.json       # OUTPUT: {category: [urls]}
├─ archive_links.txt            # OUTPUT: append-only archive of emitted links
├─ heylocate.db                 # OUTPUT: SQLite DB (seen URLs, logs, text cache, vector cache)
└─ .github/workflows/build.yml  # CI: daily run, caches, artifact commits

> **Note:** The old `seen.json`, `reasons.jsonl`, and `cache/` folder are no longer used. `init_db()` automatically migrates your old `seen.json` to SQLite on the first run.

---

## Requirements

- Python **3.11** (recommended)

Install dependencies:
pip install --upgrade pip
pip install -r requirements.txt
# (optional, local CPU-only Torch)
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.* torchvision==0.18.*

Minimal `requirements.txt`:
feedparser==6.*
trafilatura==1.*
FlagEmbedding==1.*
torch>=2.1
numpy>=1.26
aiohttp>=3.9.0
aiofiles>=23.2.0

---

## Quick Start (Local)

1) Prepare inputs in the repo root:
- `sources.txt` — your RSS/Atom feeds (one URL per line)
- `keywords.txt` — include terms (one per line)
- `stopwords.txt` — optional
- `categories.json`, `muvera_config.json`
- `blocked_domains.txt` — optional (can be empty)

2) Run:
# toggle semantic rerank: true/false
HL_SEMANTIC_RERANK=true python aggregator_links.py

3) Check outputs:
- `links.txt`, `links_by_category.json`, `archive_links.txt`
- Check internal logs and AI reasoning using any SQLite viewer (e.g., DB Browser for SQLite) by opening `heylocate.db` and browsing the `logs` table.

---

## Configuration (`muvera_config.json`)

Key fields (example defaults):
{
  "REQUIRE_KEYWORDS": true,
  "MAX_AGE_DAYS": 1,
  "REQUIRE_PUBDATE": true,
  "LIMIT_PER_RUN": 800,
  "MAX_FETCH": 30,
  "MAX_TXT_LEN": 4000,
  "CHUNK_SIZE": 700,
  "TOPK_PER_CAT": 12,
  "MV_WEIGHTS": [0.2, 0.2, 0.6],
  "THRESHOLD": 0.5,
  "UNCATEGORIZED": "Uncategorized",
  "CANONICALIZE": true,
  "LOG_REASONS": true,
  "MAX_WORKERS": 8,
  "FALLBACK_IF_EMPTY": true
}

Notes:
- **Freshness:** `MAX_AGE_DAYS` + `REQUIRE_PUBDATE` strictly keep ≤24h items (base feeds and APIs).
- **Semantic:** tune `THRESHOLD` (typical 0.45–0.55), `MAX_FETCH` (20–60), `TOPK_PER_CAT`.
- **Fallback:** if semantic keeps 0 links → automatic rule-only grouping (avoids empty outputs).

---

## GitHub Actions (Daily Run + Caches)

Workflow file: `.github/workflows/build.yml`  
Default schedule: `0 5 * * *` and `0 15 * * *` (UTC). You can also “Run workflow” manually.

**GitHub API Rate Limits:**
The workflow automatically passes `${{ secrets.GITHUB_TOKEN }}` to the script to increase GitHub Search API limits for finding new OSINT repositories.

Toggle semantic rerank in CI:
env:
  HL_SEMANTIC_RERANK: "true"   # set "false" for rule-only debugging

---

## Quality Tuning

- **Too few links**
  - Lower `THRESHOLD` to `0.45`
  - Increase `MAX_FETCH` to `40–60`
  - Loosen `keywords.txt`
  - Try `HL_SEMANTIC_RERANK=false` to validate rule-only baseline

- **Too much noise**
  - Add targeted `exclude_any` in relevant categories
  - Extend `stopwords.txt` conservatively
  - Raise `THRESHOLD` to `0.55`

---

## Troubleshooting

- **Empty `links.txt`**
  - Check logs for the fallback warning.
  - Run with `HL_SEMANTIC_RERANK=false`; if links appear, semantic threshold/text fetching is too strict.
  - Ensure feeds in `sources.txt` are reachable and publish within 24h.
  - Verify `keywords.txt` isn’t overly narrow and `stopwords.txt` isn’t cutting everything.
  - Open `heylocate.db` -> `logs` table to see what was filtered and why.

- **Process completed with exit code 1 (GitHub Actions)**
  - Usually a Git Push conflict. Ensure `git pull --rebase origin main` is added before `git push` in your `build.yml`.
  - Empty API payloads (`NoneType`) are now natively handled and won't crash the script.

- **Old links reappearing**
  - `heylocate.db` (table `seen_urls`) stores dedup memory. If you delete the DB file, older URLs may reappear (though they are still limited by the 24h freshness rule).

---

## Ethics

Use responsibly. The tool collects links and extracts text only for ranking/curation. Respect publisher policies and legal requirements in your jurisdiction.

---

## Checklist

- [ ] Provide `sources.txt`, `keywords.txt`, `categories.json`  
- [ ] Ensure `heylocate.db` is tracked by git (remove from `.gitignore` if present).
- [ ] First run with `HL_SEMANTIC_RERANK=false` → validate baseline  
- [ ] Enable semantic rerank; tune `THRESHOLD` / `MAX_FETCH`
