# Daily categorized links (no duplicates)

Outputs each day:
- `links.txt` – grouped by category (human-friendly).
- `links_by_category.json` – same data in JSON (programmatic).
- `archive_links.txt` – running history of all unique links.
- `seen.json` – store of already-seen URLs (for cross-run dedupe).

Config:
- `categories.json` – ordered list; the **first matching** category wins.
- `blocked_domains.txt` – optional domain-level blocklist (one per line).
- `keywords.txt` / `stopwords.txt` – as before; keywords are still required (`REQUIRE_KEYWORDS=True`).

Schedule:
- GitHub Actions runs once every 24h (00:00 UTC); change CRON in `.github/workflows/build.yml` if needed.
