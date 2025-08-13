import os
import feedparser, time, re, pathlib, json, hashlib, concurrent.futures
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, urljoin, quote_plus

# optional heavy deps
try:
    import trafilatura
except Exception:
    trafilatura = None
try:
    import requests
except Exception:
    requests = None

ROOT = pathlib.Path(__file__).parent

# files
SEEN_PATH   = ROOT / "seen.json"
LINKS_TXT   = ROOT / "links.txt"
LINKS_JSON  = ROOT / "links_by_category.json"
ARCHIVE_TXT = ROOT / "archive_links.txt"
CFG_PATH    = ROOT / "muvera_config.json"
CAT_PATH    = ROOT / "categories.json"
CACHE_DIR   = ROOT / "cache"
REASONS_LOG = ROOT / "reasons.jsonl"

# allow toggle from workflow via env
SEMANTIC_RERANK = os.getenv("HL_SEMANTIC_RERANK", "true").lower() in ("1","true","yes","y")

# ---------- helpers ----------
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

def norm(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def normalize_domain(d: str) -> str:
    d = (d or "").strip().lower()
    if d.startswith("www."):
        d = d[4:]
    return d

# ---------- defaults (can be overridden by muvera_config.json) ----------
CFG = {
    "REQUIRE_KEYWORDS": True,
    "LIMIT_PER_RUN": 800,
    "MAX_FETCH": 30,
    "MAX_TXT_LEN": 4000,
    "CHUNK_SIZE": 700,
    "TOPK_PER_CAT": 12,
    "MV_WEIGHTS": [0.2, 0.2, 0.6],
    "THRESHOLD": 0.50,
    "UNCATEGORIZED": "Uncategorized",
    "CANONICALIZE": True,
    "LOG_REASONS": True,
    "MAX_WORKERS": 8,
    "FALLBACK_IF_EMPTY": True,
    "DEBUG_STATS": True,
    "MAX_AGE_DAYS": 1,        # ⟵ только свежие новости (≤ 1 день)
    "REQUIRE_PUBDATE": True,  # ⟵ записи без даты отбрасывать
    "DISCOVERY": {
        "enabled": True,
        "category_name": "Outside Sources (Top 10)",
        "max_links": 10,
        "per_query_limit": 30,
        "recency": "when:1d",
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
        # Новые настройки доменов:
        "save_domains": True,
        "save_domains_file": "discovered_sources.txt"
    }
}
CFG.update(load_json(CFG_PATH, {}))
AGE_LIMIT_SEC = int(CFG.get("MAX_AGE_DAYS", 1)) * 86400

SOURCES   = read_lines(ROOT / "sources.txt")
KEYWORDS  = [s.lower() for s in read_lines(ROOT / "keywords.txt")]
STOPWORDS = [s.lower() for s in read_lines(ROOT / "stopwords.txt")]
BLOCKED_DOMAINS = {normalize_domain(d) for d in read_lines(ROOT / "blocked_domains.txt")}
CATEGORIES = load_json(CAT_PATH, [])

DISCOVERED_SOURCES_TXT = ROOT / (CFG.get("DISCOVERY", {}).get("save_domains_file", "discovered_sources.txt"))

BROAD_TOKENS = {
    "android","iphone","ios","ipados","watchos","wear os",
    "5g","lte","volte","vowifi",
    "samsung","galaxy","pixel","oneplus","xiaomi","oppo","vivo","nokia","motorola",
    "phone","smartphone","mobile","cellular"
}

# ---------- URL canonicalization ----------
_TRACKING_PARAMS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","utm_id",
                    "utm_name","utm_cid","utm_reader","gclid","fbclid","mc_cid","mc_eid","igshid"}

def canonical_url(u: str) -> str:
    """Drop tracking params, fragments, lowercase host, and try <link rel=canonical>."""
    try:
        p = urlparse(u)
        q = [(k,v) for (k,v) in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in _TRACKING_PARAMS]
        p2 = p._replace(query=urlencode(q, doseq=True), fragment="", netloc=p.netloc.lower())
        u2 = urlunparse(p2)
        if trafilatura is not None:
            html = trafilatura.fetch_url(u2, timeout=8)
            if html:
                m = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']', html, re.I)
                if m:
                    href = m.group(1)
                    if not urlparse(href).netloc:
                        href = urljoin(u2, href)
                    return href
        return u2
    except Exception:
        return u

# ---------- keyword/stopword matching ----------
def find_matches(hay: str, words):
    hits = []
    for w in words or []:
        w2 = (w or "").strip()
        if not w2:
            continue
        if (" " in w2 and w2 in hay) or (re.search(rf"\b{re.escape(w2)}\b", hay) if " " not in w2 else False):
            hits.append(w2)
    return hits

def match_topic_with_reason(title: str, summary: str):
    hay = f"{title.lower()} {summary.lower()}"
    stop_hits = find_matches(hay, STOPWORDS)
    if stop_hits:
        return False, {"stopwords": stop_hits}
    if CFG["REQUIRE_KEYWORDS"] and not KEYWORDS:
        return False, {"reason": "no_keywords"}
    narrow = [k for k in KEYWORDS if k not in BROAD_TOKENS]
    if narrow:
        hits = find_matches(hay, narrow)
        return (True, {"keywords": hits, "narrow": True}) if hits else (False, {"keywords": [], "narrow": True})
    else:
        hits = find_matches(hay, KEYWORDS)
        return (True, {"keywords": hits, "narrow": False}) if hits else (False, {"keywords": []})

# ---------- feed parsing (returns ts & has_pub) ----------
def parse_feed(url: str):
    d = feedparser.parse(url)
    items = []
    for e in d.entries:
        title = norm(getattr(e, "title", ""))
        link  = norm(getattr(e, "link", ""))
        summary = norm(getattr(e, "summary", getattr(e, "description", "")))
        pp = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
        has_pub = pp is not None
        ts = int(time.mktime(pp)) if has_pub else None
        items.append((ts, has_pub, link, title, summary))
    return items

# ---------- collecting with 1-day filter ----------
def gather():
    now = int(time.time())
    age_floor = now - AGE_LIMIT_SEC

    all_items = []
    for url in SOURCES:
        try:
            all_items.extend(parse_feed(url))
        except Exception as e:
            print("Feed error:", url, e)

    themed = []
    dropped_age = 0
    dropped_no_pub = 0

    for ts, has_pub, link, title, summary in all_items:
        if CFG.get("REQUIRE_PUBDATE", True) and not has_pub:
            dropped_no_pub += 1
            continue
        if has_pub and ts < age_floor:
            dropped_age += 1
            continue

        ok, why = match_topic_with_reason(title, summary)
        if ok:
            ts_eff = ts if ts is not None else now
            themed.append((ts_eff, link, title, summary, why))

    # newest first
    themed.sort(key=lambda x: x[0], reverse=True)

    # dedupe + domain block + canonicalize
    seen_run = set()
    unique = []
    for ts, link, title, summary, why in themed:
        if not link:
            continue
        link_c = canonical_url(link) if CFG.get("CANONICALIZE", True) else link
        domain = normalize_domain(urlparse(link_c).netloc)
        if domain in BLOCKED_DOMAINS:
            continue
        if link_c not in seen_run:
            seen_run.add(link_c)
            unique.append((ts, link_c, title, summary, domain, why))
        if len(unique) >= CFG["LIMIT_PER_RUN"]:
            break

    if CFG.get("DEBUG_STATS", True):
        from datetime import datetime as dt
        print(f"[debug] age_floor={dt.fromtimestamp(age_floor, tz=timezone.utc).isoformat()}, "
              f"dropped_age={dropped_age}, dropped_no_pub={dropped_no_pub}, kept={len(unique)}")

    return unique

# ---------- seen store ----------
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

# ---------- categorization ----------
def any_word_simple(hay: str, words):
    for w in words or []:
        if w and w in hay:
            return True
    return False

def categorize_item(title, summary, domain):
    hay = f"{title.lower()} {summary.lower()}"
    for cat in CATEGORIES:
        name = cat.get("name") or CFG["UNCATEGORIZED"]
        inc_any = [s.lower() for s in (cat.get("include_any") or [])]
        inc_all = [s.lower() for s in (cat.get("include_all") or [])]
        exc_any = [s.lower() for s in (cat.get("exclude_any") or [])]
        doms    = [normalize_domain(s) for s in (cat.get("domains") or [])]

        if doms and domain in doms:
            if exc_any and any_word_simple(hay, exc_any):
                continue
            return name

        if inc_any and not any_word_simple(hay, inc_any):
            continue
        if inc_all and not all(any_word_simple(hay, [w]) for w in inc_all):
            continue
        if exc_any and any_word_simple(hay, exc_any):
            continue
        return name
    return CFG["UNCATEGORIZED"]

# ---------- robust text fetch ----------
def _fetch_text(url: str) -> str:
    if trafilatura is not None:
        try:
            html = trafilatura.fetch_url(url, timeout=12)
            if html:
                txt = trafilatura.extract(html, include_comments=False) or ""
                if txt:
                    return txt.strip()
        except Exception:
            pass
    if requests is not None:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (HL-NewsBot)"}
            r = requests.get(url, timeout=12, headers=headers, allow_redirects=True)
            if r.ok and r.text:
                if trafilatura is not None:
                    txt = trafilatura.extract(r.text, include_comments=False) or ""
                    return (txt or "").strip()
                return r.text[:5000]
        except Exception:
            pass
    return ""

def fetch_text_cached(url: str) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    p = CACHE_DIR / (hashlib.sha1(url.encode('utf-8')).hexdigest() + ".txt")
    if p.exists():
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
    txt = _fetch_text(url)
    try:
        p.write_text(txt, encoding="utf-8")
    except Exception:
        pass
    return txt

# ---------- MUVERA-like semantic scoring (BGE-M3) ----------
_MODEL = None
def _get_model():
    global _MODEL
    from FlagEmbedding import BGEM3FlagModel
    if _MODEL is None:
        _MODEL = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
    return _MODEL

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

# ---------- reasons logging ----------
def log_reason(payload: dict):
    if not CFG.get("LOG_REASONS", True):
        return
    try:
        with open(REASONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ---------- groupers ----------
def _rule_only_grouping(items):
    grouped = {}
    for cat in [c.get("name") for c in CATEGORIES] + [CFG["UNCATEGORIZED"]]:
        grouped[cat] = []
    for ts, link, title, summary, domain, why in items:
        cat = categorize_item(title, summary, domain)
        grouped.setdefault(cat, []).append(link)
        log_reason({
            "ts": datetime.now(timezone.utc).isoformat(),
            "link": link,
            "domain": domain,
            "category": cat,
            "semantic": False,
            "keyword_hits": (why.get("keywords") if isinstance(why, dict) else None)
        })
    return grouped

def semantic_filter_and_rank(items):
    if not SEMANTIC_RERANK or trafilatura is None:
        return _rule_only_grouping(items)

    mv_weights = CFG["MV_WEIGHTS"]; threshold = CFG["THRESHOLD"]
    max_fetch  = CFG["MAX_FETCH"];   max_txt   = CFG["MAX_TXT_LEN"]
    chunk_size = CFG["CHUNK_SIZE"];  topk      = CFG["TOPK_PER_CAT"]
    max_workers = int(CFG.get("MAX_WORKERS", 8))

    cat_queries = [(c.get("name") or CFG["UNCATEGORIZED"], _build_query_for_category(c)) for c in CATEGORIES]
    cat_queries.append((CFG["UNCATEGORIZED"], "mobile security telecom lookup"))

    to_score = items[:max_fetch]
    urls = [link for _, link, *_ in to_score]
    texts = {}
    empty_texts = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for url, txt in zip(urls, ex.map(fetch_text_cached, urls)):
            if not txt:
                empty_texts += 1
            texts[url] = txt

    kept = {name: [] for name, _ in cat_queries}
    for (ts, link, title, summary, domain, why) in to_score:
        text = texts.get(link) or ""
        if not text:
            continue
        for name, q in cat_queries:
            try:
                score = mv_score(q, text, mv_weights, max_txt, chunk_size)
            except Exception as e:
                print("Semantic scoring error:", e)
                break
        # apply threshold
            if score >= threshold:
                kept[name].append((score, link))
                log_reason({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "link": link,
                    "domain": domain,
                    "category": name,
                    "semantic": True,
                    "score": round(float(score), 4),
                    "threshold": threshold,
                    "query": q,
                    "keyword_hits": (why.get("keywords") if isinstance(why, dict) else None)
                })
                break

    grouped = {name: [u for s, u in sorted(vals, key=lambda x: x[0], reverse=True)[:topk]]
               for name, vals in kept.items()}

    total = sum(len(v) for v in grouped.values())
    if CFG.get("DEBUG_STATS", True):
        print(f"[debug] items={len(items)}, to_score={len(to_score)}, empty_texts={empty_texts}, kept_total={total}")

    if total == 0 and CFG.get("FALLBACK_IF_EMPTY", True):
        print("[warn] semantic produced 0 links; falling back to rule-only grouping")
        return _rule_only_grouping(items)

    return grouped

# ---------- discovery: outside sources with same 1-day filter ----------
def _known_source_domains():
    dset = set()
    for s in SOURCES:
        try:
            dset.add(normalize_domain(urlparse(s).netloc))
        except Exception:
            pass
    return dset

def _gnews_search_feed(query: str) -> str:
    recency = CFG["DISCOVERY"].get("recency", "when:1d")
    hl = CFG["DISCOVERY"].get("hl", "en-US")
    gl = CFG["DISCOVERY"].get("gl", "US")
    ceid = CFG["DISCOVERY"].get("ceid", "US:en")
    q = quote_plus(f"{query} {recency}")
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"

def _read_discovered_domains(path: pathlib.Path) -> set:
    if not path.exists():
        return set()
    try:
        return {normalize_domain(x) for x in read_lines(path)}
    except Exception:
        return set()

def _write_discovered_domains(path: pathlib.Path, domains: set):
    if not domains:
        return
    existing = _read_discovered_domains(path)
    merged = sorted({*existing, *{normalize_domain(d) for d in domains}} - {""})
    try:
        path.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")
    except Exception as e:
        print("Discovered domains write error:", e)

def discover_outside(grouped_existing, seen_dict, already_links):
    if not CFG.get("DISCOVERY", {}).get("enabled", False):
        return []

    now = int(time.time())
    age_floor = now - AGE_LIMIT_SEC
    known = _known_source_domains()
    cat_queries = [(_build_query_for_category(c)) for c in CATEGORIES] or ["mobile security telecom lookup"]

    per_query_limit = int(CFG["DISCOVERY"].get("per_query_limit", 30))
    candidates = []
    for q in cat_queries:
        feed_url = _gnews_search_feed(q)
        try:
            for ts, has_pub, link, title, summary in parse_feed(feed_url):
                if CFG.get("REQUIRE_PUBDATE", True) and not has_pub:
                    continue
                if has_pub and ts < age_floor:
                    continue
                link_c = canonical_url(link)
                dom = normalize_domain(urlparse(link_c).netloc)
                if dom in known or dom in BLOCKED_DOMAINS:
                    continue
                if (link_c in already_links) or (link_c in seen_dict):
                    continue
                ok, _ = match_topic_with_reason(title, summary)
                if not ok:
                    continue
                candidates.append((ts or now, link_c, title, summary, dom))
                if len(candidates) >= per_query_limit * len(cat_queries):
                    break
        except Exception as e:
            print("Discovery feed error:", feed_url, e)

    if not candidates:
        return []

    mv_weights = CFG["MV_WEIGHTS"]; threshold = CFG["THRESHOLD"]
    max_txt = CFG["MAX_TXT_LEN"]; chunk_size = CFG["CHUNK_SIZE"]
    max_workers = int(CFG.get("MAX_WORKERS", 8))

    urls = [link for _, link, *_ in candidates]
    texts = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for url, txt in zip(urls, ex.map(fetch_text_cached, urls)):
            texts[url] = txt

    scored = []
    for (ts, link, title, summary, dom) in candidates:
        text = texts.get(link) or ""
        if not text:
            continue
        best = 0.0; best_q = None
        for q in cat_queries:
            try:
                s = mv_score(q, text, mv_weights, max_txt, chunk_size)
            except Exception:
                s = 0.0
            if s > best:
                best = s; best_q = q
        if best >= threshold:
            scored.append((best, link, dom, best_q))

    scored.sort(key=lambda x: x[0], reverse=True)
    lim = int(CFG["DISCOVERY"].get("max_links", 10))
    out_links = []
    out_domains = set()
    for s, link, dom, best_q in scored[:lim]:
        out_links.append(link)
        out_domains.add(dom)
        log_reason({
            "ts": datetime.now(timezone.utc).isoformat(),
            "link": link,
            "domain": dom,
            "category": CFG["DISCOVERY"].get("category_name", "Outside Sources"),
            "semantic": True,
            "score": round(float(s), 4),
            "threshold": threshold,
            "query": best_q,
            "outside": True
        })

    # save discovered domains if enabled
    if CFG.get("DISCOVERY", {}).get("save_domains", True):
        _write_discovered_domains(DISCOVERED_SOURCES_TXT, out_domains)

    return out_links

# ---------- outputs ----------
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

# ---------- main ----------
if __name__ == "__main__":
    CACHE_DIR.mkdir(exist_ok=True)
    items = gather()  # [(ts, link, title, summary, domain, why)]
    seen = load_seen()
    now_iso = datetime.now(timezone.utc).isoformat()

    new_items = [it for it in items if it[1] not in seen]

    grouped = semantic_filter_and_rank(new_items)

    # discovery category (+ auto-save discovered domains)
    try:
        already = set([u for urls in grouped.values() for u in urls])
        discovery_links = discover_outside(grouped, seen, already)
        if discovery_links:
            grouped[CFG["DISCOVERY"].get("category_name", "Outside Sources (Top 10)")] = discovery_links
    except Exception as e:
        print("Discovery error:", e)

    # outputs
    write_outputs(grouped)

    # update seen + archive with all new links (incl. discovery)
    new_links = set([link for _, link, *_ in new_items])
    for urls in grouped.values():
        for link in urls:
            new_links.add(link)
    for link in new_links:
        seen[link] = now_iso
    seen = prune_seen(seen, keep=50000)
    save_seen(seen)
    append_archive(sorted(new_links))

    total_new = sum(len(v) for v in grouped.values())
    print(f"[OK] New categorized links (incl. discovery): {total_new}")
