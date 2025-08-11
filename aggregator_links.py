import feedparser, time, re, pathlib, json
from datetime import datetime, timezone
from urllib.parse import urlparse

ROOT = pathlib.Path(__file__).parent
SEEN_PATH = ROOT / "seen.json"
LINKS_TXT = ROOT / "links.txt"                    # now grouped by category
LINKS_JSON = ROOT / "links_by_category.json"      # machine-friendly output
ARCHIVE_PATH = ROOT / "archive_links.txt"

# ---- Config ----
REQUIRE_KEYWORDS = True  # if True and keywords list is empty -> skip everything
LIMIT = 1000             # safety cap per run
UNCAT = "Uncategorized"

# Broad tokens: allowed only as context; alone they should not match
BROAD_TOKENS = {
    "android","iphone","ios","ipados","watchos","wear os",
    "5g","lte","volte","vowifi",
    "samsung","galaxy","pixel","oneplus","xiaomi","oppo","vivo","nokia","motorola",
    "phone","smartphone","mobile","cellular"
}

def read_lines(path: pathlib.Path):
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")]

def load_json(path: pathlib.Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback

SOURCES   = read_lines(ROOT / "sources.txt")
KEYWORDS  = [s.lower() for s in read_lines(ROOT / "keywords.txt")]
STOPWORDS = [s.lower() for s in read_lines(ROOT / "stopwords.txt")]
BLOCKED_DOMAINS = {d.lower() for d in read_lines(ROOT / "blocked_domains.txt")}  # optional
CATEGORIES = load_json(ROOT / "categories.json", [])  # list of {"name":..., "include_any":[...], "include_all":[...], "exclude_any":[...], "domains":[...]}

def norm(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def any_word(hay: str, words):
    """Phrases (with spaces) -> substring; single tokens -> word-boundary regex."""
    for w in words or []:
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
    if STOPWORDS and any_word(hay, STOPWORDS):
        return False
    if REQUIRE_KEYWORDS and not KEYWORDS:
        return False
    narrow = [k for k in KEYWORDS if k not in BROAD_TOKENS]
    if narrow and any_word(hay, narrow):
        return True
    if not narrow and KEYWORDS and any_word(hay, KEYWORDS):
        return True
    return False

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
    themed.sort(key=lambda x: x[0], reverse=True)  # newest first
    seen_run = set()
    unique = []
    for ts, link, title, summary in themed:
        if not link:
            continue
        domain = urlparse(link).netloc.lower()
        if domain in BLOCKED_DOMAINS:
            continue
        if link not in seen_run:
            seen_run.add(link)
            unique.append((ts, link, title, summary, domain))
        if len(unique) >= LIMIT:
            break
    return unique

def load_seen():
    if not SEEN_PATH.exists():
        return {}
    try:
        data = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            now = datetime.now(timezone.utc).isoformat()
            return {u: now for u in data}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_seen(seen: dict):
    SEEN_PATH.write_text(json.dumps(seen, ensure_ascii=False, indent=2), encoding="utf-8")

def prune_seen(seen: dict, keep=50000):
    if len(seen) <= keep:
        return seen
    def ts_or_min(iso):
        try:
            return datetime.fromisoformat(iso.replace("Z",""))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)
    items = sorted(seen.items(), key=lambda kv: ts_or_min(kv[1]), reverse=True)[:keep]
    return dict(items)

def categorize_item(link, title, summary, domain):
    """Return first-matching category per the order in categories.json.
       If none matches, return UNCAT."""
    hay = f"{title.lower()} {summary.lower()}"
    for cat in CATEGORIES:
        name = cat.get("name") or UNCAT
        inc_any = [s.lower() for s in (cat.get("include_any") or [])]
        inc_all = [s.lower() for s in (cat.get("include_all") or [])]
        exc_any = [s.lower() for s in (cat.get("exclude_any") or [])]
        doms    = [s.lower() for s in (cat.get("domains") or [])]

        if doms and domain in doms:
            # domain match can still be vetoed by exclude
            if exc_any and any_word(hay, exc_any):
                continue
            return name

        if inc_any and not any_word(hay, inc_any):
            continue
        if inc_all and not all(any_word(hay, [w]) for w in inc_all):
            continue
        if exc_any and any_word(hay, exc_any):
            continue
        return name
    return UNCAT

def write_outputs(grouped):
    # 1) links_by_category.json
    LINKS_JSON.write_text(json.dumps(grouped, ensure_ascii=False, indent=2), encoding="utf-8")
    # 2) links.txt (readable sections)
    lines = []
    for cat, urls in grouped.items():
        if not urls:
            continue
        lines.append(f"## {cat}")
        lines.extend(urls)
        lines.append("")  # blank line between sections
    LINKS_TXT.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

def append_archive(urls):
    if not urls:
        return
    with open(ARCHIVE_PATH, "a", encoding="utf-8") as f:
        for link in urls:
            f.write(link + "\n")

if __name__ == "__main__":
    items = gather()  # [(ts, link, title, summary, domain)]
    seen = load_seen()
    now_iso = datetime.now(timezone.utc).isoformat()

    # only new links vs. seen.json
    new_items = [it for it in items if it[1] not in seen]

    # group by category (preserving order from categories.json)
    cat_order = [c.get("name") for c in CATEGORIES] + [UNCAT]
    grouped = {name: [] for name in cat_order}
    for ts, link, title, summary, domain in new_items:
        cat = categorize_item(link, title, summary, domain)
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(link)

    # write outputs
    write_outputs(grouped)

    # update seen + archive
    for _, link, *_ in new_items:
        seen[link] = now_iso
    seen = prune_seen(seen, keep=50000)
    save_seen(seen)
    append_archive([link for _, link, *_ in new_items])

    total_new = sum(len(v) for v in grouped.values())
    print(f"New categorized links: {total_new}; categories: {len([k for k,v in grouped.items() if v])}")
