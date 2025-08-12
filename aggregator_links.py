import feedparser, time, re, pathlib, json
from datetime import datetime, timezone
from urllib.parse import urlparse

try:
    import trafilatura  # used only when semantic rerank is enabled
except Exception:
    trafilatura = None

ROOT = pathlib.Path(__file__).parent

SEEN_PATH   = ROOT / "seen.json"
LINKS_TXT   = ROOT / "links.txt"
LINKS_JSON  = ROOT / "links_by_category.json"
ARCHIVE_TXT = ROOT / "archive_links.txt"
CFG_PATH    = ROOT / "muvera_config.json"
CAT_PATH    = ROOT / "categories.json"

SEMANTIC_RERANK = True  # set False to disable semantic step

CFG = {
    "REQUIRE_KEYWORDS": True,
    "LIMIT_PER_RUN": 800,
    "MAX_FETCH": 40,       # tuned for GitHub Actions CPU/time
    "MAX_TXT_LEN": 8000,
    "CHUNK_SIZE": 700,
    "TOPK_PER_CAT": 12,
    "MV_WEIGHTS": [0.2, 0.2, 0.6],
    "THRESHOLD": 0.60,     # slightly stricter than default
    "UNCATEGORIZED": "Uncategorized"
}

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

# allow overriding via file
CFG.update(load_json(CFG_PATH, {}))

SOURCES   = read_lines(ROOT / "sources.txt")
KEYWORDS  = [s.lower() for s in read_lines(ROOT / "keywords.txt")]
STOPWORDS = [s.lower() for s in read_lines(ROOT / "stopwords.txt")]
BLOCKED_DOMAINS = {d.lower() for d in read_lines(ROOT / "blocked_domains.txt")}
CATEGORIES = load_json(CAT_PATH, [])

def norm(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def any_word(hay: str, words):
    for w in words or []:
        w = w.strip()
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
    if CFG["REQUIRE_KEYWORDS"] and not KEYWORDS:
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
    themed.sort(key=lambda x: x[0], reverse=True)
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
        if len(unique) >= CFG["LIMIT_PER_RUN"]:
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

def categorize_item(title, summary, domain):
    hay = f"{title.lower()} {summary.lower()}"
    for cat in CATEGORIES:
        name = cat.get("name") or CFG["UNCATEGORIZED"]
        inc_any = [s.lower() for s in (cat.get("include_any") or [])]
        inc_all = [s.lower() for s in (cat.get("include_all") or [])]
        exc_any = [s.lower() for s in (cat.get("exclude_any") or [])]
        doms    = [s.lower() for s in (cat.get("domains") or [])]
        if doms and domain in doms:
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
    return CFG["UNCATEGORIZED"]

# === Semantic rerank (BGE-M3) ===
_MODEL = None
def _get_model():
    global _MODEL
    from FlagEmbedding import BGEM3FlagModel
    if _MODEL is None:
        _MODEL = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
    return _MODEL

def _fetch_text(url: str) -> str:
    if trafilatura is None:
        return ""
    try:
        html = trafilatura.fetch_url(url, timeout=12)
        if not html:
            return ""
        txt = trafilatura.extract(html, include_comments=False) or ""
        return txt.strip()
    except Exception:
        return ""

def _chunks(s: str, n: int):
    for i in range(0, len(s), n):
        yield s[i:i+n]

def _build_query_for_category(cat: dict) -> str:
    inc_any = cat.get("include_any") or []
    inc_all = cat.get("include_all") or []
    parts = (inc_all[:3] if inc_all else []) + (inc_any[:5] if inc_any else [])
    return " ; ".join(parts) or "mobile security telecom lookup"

def mv_score(query: str, doc_text: str, mv_weights, max_txt, chunk_size) -> float:
    m = _get_model()
    q_out = m.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=True)
    best = 0.0
    for piece in _chunks(doc_text[:max_txt], chunk_size):
        d_out = m.encode([piece], return_dense=True, return_sparse=True, return_colbert_vecs=True)
        d = float((q_out['dense_vecs'] @ d_out['dense_vecs'].T)[0,0])
        s = float(m.compute_lexical_matching_score(q_out['lexical_weights'][0], d_out['lexical_weights'][0]))
        c = float(m.colbert_score(q_out['colbert_vecs'][0], d_out['colbert_vecs'][0]))
        hybrid = mv_weights[0]*d + mv_weights[1]*s + mv_weights[2]*c
        if hybrid > best:
            best = hybrid
    return best

def semantic_filter_and_rank(items):
    # Fallback to rule-only if disabled or trafilatura unavailable
    if not SEMANTIC_RERANK or trafilatura is None:
        grouped = {}
        for cat in [c.get("name") for c in CATEGORIES] + [CFG["UNCATEGORIZED"]]:
            grouped[cat] = []
        for ts, link, title, summary, domain in items:
            cat = categorize_item(title, summary, domain)
            grouped.setdefault(cat, []).append(link)
        return grouped

    mv_weights = CFG["MV_WEIGHTS"]
    threshold  = CFG["THRESHOLD"]
    max_fetch  = CFG["MAX_FETCH"]
    max_txt    = CFG["MAX_TXT_LEN"]
    chunk_size = CFG["CHUNK_SIZE"]
    topk       = CFG["TOPK_PER_CAT"]

    cat_queries = [(c.get("name") or CFG["UNCATEGORIZED"], _build_query_for_category(c)) for c in CATEGORIES]
    cat_queries.append((CFG["UNCATEGORIZED"], "mobile security telecom lookup"))

    kept = {name: [] for name, _ in cat_queries}
    fetched = 0
    for ts, link, title, summary, domain in items:
        if fetched >= max_fetch:
            break
        text = _fetch_text(link)
        if not text:
            continue
        fetched += 1
        for name, q in cat_queries:
            try:
                score = mv_score(q, text, mv_weights, max_txt, chunk_size)
            except Exception as e:
                print("Semantic scoring error:", e)
                # fallback to rule-only grouping
                grouped = {}
                for cat in [c.get("name") for c in CATEGORIES] + [CFG["UNCATEGORIZED"]]:
                    grouped[cat] = []
                for ts2, link2, title2, summary2, domain2 in items:
                    cat2 = categorize_item(title2, summary2, domain2)
                    grouped.setdefault(cat2, []).append(link2)
                return grouped
            if score >= threshold:
                kept[name].append((score, link))
                break

    grouped = {name: [u for s, u in sorted(vals, key=lambda x: x[0], reverse=True)[:topk]]
               for name, vals in kept.items()}
    return grouped

def write_outputs(grouped):
    LINKS_JSON.write_text(json.dumps(grouped, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = []
    for cat, urls in grouped.items():
        if not urls:
            continue
        lines.append(f"## {cat}")
        lines.extend(urls)
        lines.append("")
    LINKS_TXT.write_text(("\n".join(lines).strip() + "\n") if lines else "", encoding="utf-8")

def append_archive(urls):
    if not urls:
        return
    with open(ARCHIVE_TXT, "a", encoding="utf-8") as f:
        for link in urls:
            f.write(link + "\n")

if __name__ == "__main__":
    items = gather()
    seen = load_seen()
    now_iso = datetime.now(timezone.utc).isoformat()

    new_items = [it for it in items if it[1] not in seen]

    grouped = semantic_filter_and_rank(new_items)

    write_outputs(grouped)

    new_links = [link for _, link, *_ in new_items]
    for link in new_links:
        seen[link] = now_iso
    seen = prune_seen(seen, keep=50000)
    save_seen(seen)
    append_archive(new_links)

    total_new = sum(len(v) for v in grouped.values())
    print(f"[OK] New categorized links: {total_new}; semantic_fetch={min(len(new_items), CFG['MAX_FETCH']) if SEMANTIC_RERANK else 0}")
