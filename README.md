# News Links Aggregator (Free, RSS-based)

Collects **links only** to fresh, topic-specific materials from RSS feeds. No paid APIs.

## How it works
- Reads the RSS list from `sources.txt`.
- Filters by keywords from `keywords.txt` (if the file is empty — it takes everything).
- Excludes by `stopwords.txt` (optional).
- Sorts by time and saves a **clean list of links** to `links.txt` (one URL per line, up to 200).

## Quick start (locally)
1) Install Python 3.10+.
2) `pip install -r requirements.txt`
3) Edit `sources.txt` and (optionally) `keywords.txt`.
4) Run: `python aggregator_links.py`
5) Done: check `links.txt`.

## GitHub Actions (auto-run)
The workflow in `.github/workflows/build.yml` updates `links.txt` every 30 minutes.
