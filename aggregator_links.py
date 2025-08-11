import feedparser, time, re, pathlib
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).parent

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
LIMIT     = 200

def norm(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def match_topic(title: str, summary: str) -> bool:
    hay = f"{title.lower()} {summary.lower()}"
    if STOPWORDS and any(sw in hay for sw in STOPWORDS):
        return False
    if KEYWORDS:
        return any(kw in hay for kw in KEYWORDS)
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
    themed.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    unique = []
    for ts, link, title, summary in themed:
        if link and link not in seen:
            seen.add(link)
            unique.append((ts, link))
        if len(unique) >= LIMIT:
            break
    return unique

def write_links(items):
    with open(ROOT / "links.txt", "w", encoding="utf-8") as f:
        for _, link in items:
            f.write(link + "\n")

if __name__ == "__main__":
    items = gather()
    write_links(items)
    print(f"Generated links.txt with {len(items)} links")
