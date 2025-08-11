import feedparser, time, re, pathlib, json
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).parent
SEEN_PATH = ROOT / "seen.json"
LINKS_PATH = ROOT / "links.txt"
ARCHIVE_PATH = ROOT / "archive_links.txt"

def read_lines(path: pathlib.Path):
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines

SOURCES   = read_lines(ROOT / "sources.txt")
KEYWORDS  = [s.lower() for s in read_lines(ROOT / "keywords.txt")]
STOPWORDS = [s.lower() for s in read_lines(ROOT / "stopwords.txt")]
LIMIT     = 1000  # safety cap per run

def norm(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def any_word(hay: str, words):
    # stricter matching for single words; phrases with spaces are matched as substrings
    for w in words:
        if not w:
            continue
        if " " in w:
            if w in hay:
                return True
        else:
            if re.search(rf"\b{re.escape(w)}\b", hay):
                return True
    return False

def match_topic(title: str, summary: str) -> bool:
    hay = f"{title.lower()} {summary.lower()}"
    # stopwords win
    if STOPWORDS and any_word(hay, STOPWORDS):
        return False
    # keywords optional: if present, require at least one
    if KEYWORDS:
        return any_word(hay, KEYWORDS)
    return True

def parse_feed(url: str):
    d = feedparser.parse(url)
    items = []
    for e in d.entries:
        title = norm(getattr(e, "title", ""))
        link  = norm(getattr(e, "link", ""))
        summary = norm(getattr(e, "summary", getattr(e, "description", "")))
        published_parsed = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
        ts = int(time.mktime(published_parsed)) if published_parsed else int(time.time())
        items.append((ts, link, title, summary))
    return items

def gather():
    all_items = []
    for url in SOURCES:
        try:
            all_items.extend(parse_feed(url))
        except Exception as e:
            print("Feed error:", url, e)
    themed = [it for it in all_items if match_topic(it[2], it[3])]
    # newest first
    themed.sort(key=lambda x: x[0], reverse=True)
    # dedupe within run by URL
    seen_run = set()
    unique = []
    for ts, link, title, summary in themed:
        if link and link not in seen_run:
            seen_run.add(link)
            unique.append((ts, link))
        if len(unique) >= LIMIT:
            break
    return unique

def load_seen():
    if not SEEN_PATH.exists():
        return {}
    try:
        data = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        # data format: {url: iso_ts}
        if isinstance(data, list):  # legacy support
            return {u: datetime.now(timezone.utc).isoformat() for u in data}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_seen(seen: dict):
    SEEN_PATH.write_text(json.dumps(seen, ensure_ascii=False, indent=2), encoding="utf-8")

def prune_seen(seen: dict, keep=50000):
    if len(seen) <= keep:
        return seen
    # keep newest by timestamp
    def ts_or_min(iso):
        try:
            return datetime.fromisoformat(iso.replace("Z",""))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)
    items = sorted(seen.items(), key=lambda kv: ts_or_min(kv[1]), reverse=True)[:keep]
    return dict(items)

def write_links(urls):
    # overwrite with only NEW links from this run
    with open(LINKS_PATH, "w", encoding="utf-8") as f:
        for link in urls:
            f.write(link + "\n")

def append_archive(urls):
    if not urls:
        return
    with open(ARCHIVE_PATH, "a", encoding="utf-8") as f:
        for link in urls:
            f.write(link + "\n")

if __name__ == "__main__":
    collected = gather()                           # [(ts, link)]
    seen = load_seen()                             # {link: iso_ts}
    now_iso = datetime.now(timezone.utc).isoformat()

    # Filter out links already collected in previous runs
    new_links = [link for _, link in collected if link not in seen]

    # Output ONLY new links from the last 24h run
    write_links(new_links)

    # Update seen + archive
    for link in new_links:
        seen[link] = now_iso
    seen = prune_seen(seen, keep=50000)
    save_seen(seen)
    append_archive(new_links)

    print(f"Found {len(collected)} candidates; new: {len(new_links)}; seen total: {len(seen)}")
