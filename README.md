# News Links Aggregator
_MUVERA-style multi-vector rerank · daily discovery · strict freshness (≤24h)_

This tool collects **fresh, relevant links** across telecom, caller ID/robocalls, parental control, people/address/email lookup, OSINT, and related niches.  
It reads your RSS feeds, filters by **keywords/stopwords**, enforces **≤ 24 hours** freshness, optionally **reranks semantically** (MUVERA-like hybrid with `BAAI/bge-m3`), routes items into **categories**, and adds a separate **discovery** bucket from Google News RSS. Results are plain text/JSON for easy manual review.

---

## Features

- **Freshness gate:** only items **published within the last 24 hours** (`MAX_AGE_DAYS=1`, `REQUIRE_PUBDATE=true`).
- **Keyword prefilter + stopword filter** before heavy scoring.
- **MUVERA-style semantic rerank (optional):** hybrid dense + sparse + ColBERT using `BAAI/bge-m3`.
- **Categorization:** rule-based routing via `categories.json` (`include_any`, `include_all`, `exclude_any`, optional `domains`).
- **Discovery (outside your sources):** queries Google News RSS per category; respects freshness, dedup, and blocked domains.
- **Auto-save discovered domains:** writes unique domains to `discovered_sources.txt` for later promotion into `sources.txt`.
- **Canonicalization & dedup:** strips common tracking params, normalizes host, attempts `<link rel="canonical">`.
- **Resilient text fetch:** `trafilatura` + `requests` with on-disk cache.
- **GitHub Actions ready:** pip/HF caches, CPU Torch wheels, daily schedule, optional Slack notifications.

---

## Repository Layout

```
.
├─ aggregator_links.py          # main script
├─ muvera_config.json           # engine configuration (freshness, thresholds, discovery)
├─ categories.json              # topic routing rules
├─ sources.txt                  # RSS/Atom feeds (one URL per line)
├─ keywords.txt                 # positive terms (one per line)
├─ stopwords.txt                # optional global negative terms
├─ blocked_domains.txt          # optional domain blocklist (can be empty)
│
├─ links.txt                    # OUTPUT: human-readable, grouped by category
├─ links_by_category.json       # OUTPUT: {category: [urls]}
├─ archive_links.txt            # OUTPUT: append-only archive of emitted links
├─ discovered_sources.txt       # OUTPUT: unique domains found via discovery
├─ reasons.jsonl                # OUTPUT: per-link reasoning/log (JSONL)
├─ seen.json                    # de-dup memory (url → ISO timestamp)
├─ cache/                       # article text cache (sha1.txt)
└─ .github/workflows/build.yml  # CI: daily run, caches, optional Slack
```

> Add `cache/.gitignore`:
```
*
!.gitignore
```

---

## Requirements

- Python **3.11** (recommended)

Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
# (optional, local CPU-only Torch)
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.* torchvision==0.18.*
```

Minimal `requirements.txt`:
```
feedparser
trafilatura
FlagEmbedding
torch
numpy
requests
```

---

## Quick Start (Local)

1) Prepare inputs in the repo root:
- `sources.txt` — your RSS/Atom feeds (one URL per line)
- `keywords.txt` — include terms (one per line)
- `stopwords.txt` — optional
- `categories.json`, `muvera_config.json`
- `blocked_domains.txt` — optional (can be empty)

2) Run:
```bash
# toggle semantic rerank: true/false
HL_SEMANTIC_RERANK=true python aggregator_links.py
```

3) Check outputs:
- `links.txt`, `links_by_category.json`, `archive_links.txt`
- `reasons.jsonl` — why a link was kept (scores, keywords, query, category)
- `discovered_sources.txt` — new external domains (review & promote to `sources.txt` if good)

---

## Configuration (`muvera_config.json`)

Key fields (example defaults):
```json
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
  "FALLBACK_IF_EMPTY": true,
  "DEBUG_STATS": true,
  "DISCOVERY": {
    "enabled": true,
    "category_name": "Outside Sources (Top 10)",
    "max_links": 10,
    "per_query_limit": 30,
    "recency": "when:1d",
    "hl": "en-US",
    "gl": "US",
    "ceid": "US:en",
    "save_domains": true,
    "save_domains_file": "discovered_sources.txt"
  }
}
```

Notes:
- **Freshness:** `MAX_AGE_DAYS` + `REQUIRE_PUBDATE` strictly keep ≤24h items (base feeds and discovery).
- **Semantic:** tune `THRESHOLD` (typical 0.45–0.55), `MAX_FETCH` (20–60), `TOPK_PER_CAT`.
- **Fallback:** if semantic keeps 0 links → automatic rule-only grouping (avoids empty outputs).
- **Discovery:** queries are built from `categories.json`, results exclude domains in `sources.txt` and `blocked_domains.txt`, and unique domains are saved to `discovered_sources.txt`.

---

## GitHub Actions (Daily Run + Caches + Slack)

Workflow file: `.github/workflows/build.yml`  
Default schedule: `0 3 * * *` (03:00 UTC). You can also “Run workflow” manually.

Toggle semantic rerank in CI:
```yaml
env:
  HL_SEMANTIC_RERANK: "true"   # set "false" for rule-only debugging
```

**Slack notifications (optional)**
1) Create a Slack **Incoming Webhook** and add repo secret `SLACK_WEBHOOK_URL`.  
2) The workflow posts job status, total links, and top categories after each run.

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

- **Discovery too broad/narrow**
  - Tweak `DISCOVERY.max_links` and `per_query_limit`
  - Review `discovered_sources.txt` and promote good domains to `sources.txt`
  - Add unwanted domains to `blocked_domains.txt`

---

## Troubleshooting

- **Empty `links.txt`**
  - Check logs for `[debug]` and the fallback warning
  - Run with `HL_SEMANTIC_RERANK=false`; if links appear, semantic threshold/text fetching is too strict
  - Ensure feeds in `sources.txt` are reachable and publish within 24h
  - Verify `keywords.txt` isn’t overly narrow and `stopwords.txt` isn’t cutting everything
  - Inspect `reasons.jsonl` to see what was filtered and why

- **Slow first run**
  - HF model and Torch are cached in CI; re-runs are faster
  - Use CPU Torch wheels locally if needed (see Requirements)

- **Old links reappearing**
  - `seen.json` stores dedup memory; removing/renaming it may allow older URLs to reappear (still limited by 24h freshness)

---

## Ethics

Use responsibly. The tool collects links and extracts text only for ranking/curation. Respect publisher policies and legal requirements in your jurisdiction.

---

## Checklist

- [ ] Provide `sources.txt`, `keywords.txt`, `categories.json`  
- [ ] (Optional) prepare `stopwords.txt`, `blocked_domains.txt`  
- [ ] First run with `HL_SEMANTIC_RERANK=false` → validate baseline  
- [ ] Enable semantic rerank; tune `THRESHOLD` / `MAX_FETCH`  
- [ ] Review `discovered_sources.txt` and promote good domains to `sources.txt`
