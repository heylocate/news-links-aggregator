# Discovery category enabled

Adds a special category **"Outside Sources (Top 10)"**:
- Pulls candidates from Google News RSS searches built from your category queries.
- Keeps only links from domains **not present** in `sources.txt`.
- Applies the **same semantic scoring** and threshold as other categories.
- Writes reasons to `reasons.jsonl` (with `outside:true`).

Tune in `muvera_config.json` → `"DISCOVERY"`:
- `enabled` (true/false), `max_links`, `per_query_limit`, `recency` (e.g., `when:1d`), `hl/gl/ceid` for locale.
