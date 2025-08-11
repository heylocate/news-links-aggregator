# Daily unique links

This version:
- runs **once every 24h** via GitHub Actions (00:00 UTC),
- writes **links.txt** containing only the **new** (never-before-seen) links,
- maintains a persistent **seen.json** (for dedup across runs),
- appends all unique links to **archive_links.txt** (running history).

If you want a different time, change the CRON in `.github/workflows/build.yml`.
