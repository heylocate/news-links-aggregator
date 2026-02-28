import os
import json
import time
import re
import pathlib
import hashlib
import logging
import concurrent.futures
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable, Set, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, urljoin, quote_plus

import feedparser
import requests

try:
    import trafilatura
except Exception:
    trafilatura = None

ROOT = pathlib.Path(__file__).parent

SEEN_PATH = ROOT / "seen.json"
LINKS_TXT = ROOT / "links.txt"
LINKS_JSON = ROOT / "links_by_category.json"
LINKS_FULL_JSON = ROOT / "links_by_category_full.json"
ARCHIVE_TXT = ROOT / "archive_links.txt"
CFG_PATH = ROOT / "muvera_config.json"
CAT_PATH = ROOT / "categories.json"
CACHE_DIR = ROOT / "cache"
REASONS_LOG = ROOT / "reasons.jsonl"

SEMANTIC_RERANK = os.getenv("HL_SEMANTIC_RERANK", "true").lower() in ("1", "true", "yes", "y")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# ---------------------------------------------------------------------------
# Logging & Helpers
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("heylocate.news_aggregator")
if not LOGGER.handlers:
    log_level = getattr(logging, os.getenv("HL_LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    LOGGER.setLevel(log_level)

def read_lines(path: pathlib.Path) -> List[str]:
    if not path.exists(): return []
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.strip().startswith("#")]

def load_json(path: pathlib.Path, fallback):
    if not path.exists(): return fallback
    try: return json.loads(path.read_text(encoding="utf-8"))
    except: return fallback

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip() if text else ""

def normalize_domain(d: str) -> str:
    d = (d or "").strip().lower()
    return d[4:] if d.startswith("www.") else d

def atomic_write_text(path: pathlib.Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)

# ---------------------------------------------------------------------------
# Config & Keyword Matching
# ---------------------------------------------------------------------------

CFG = {"REQUIRE_KEYWORDS": True, "LIMIT_PER_RUN": 800, "MAX_FETCH": 30, "MAX_TXT_LEN": 4000, "CHUNK_SIZE": 700, "TOPK_PER_CAT": 12, "MV_WEIGHTS": [0.2, 0.2, 0.6], "THRESHOLD": 0.50, "UNCATEGORIZED": "Uncategorized", "CANONICALIZE": True, "LOG_REASONS": True, "MAX_WORKERS": 8, "MAX_AGE_DAYS": 1, "REQUIRE_PUBDATE": True}
CFG.update(load_json(CFG_PATH, {}))
AGE_LIMIT_SEC = int(CFG.get("MAX_AGE_DAYS", 1)) * 86400

SOURCES = read_lines(ROOT / "sources.txt")
KEYWORDS = [s.lower() for s in read_lines(ROOT / "keywords.txt")]
STOPWORDS = [s.lower() for s in read_lines(ROOT / "stopwords.txt")]
BLOCKED_DOMAINS = {normalize_domain(d) for d in read_lines(ROOT / "blocked_domains.txt")}
CATEGORIES = load_json(CAT_PATH, [])

BROAD_TOKENS = {"android", "iphone", "ios", "ipados", "watchos", "wear os", "5g", "lte", "volte", "vowifi", "samsung", "galaxy", "pixel", "oneplus", "xiaomi", "oppo", "vivo", "nokia", "motorola", "phone", "smartphone", "mobile", "cellular"}
_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id", "utm_name", "utm_cid", "utm_reader", "gclid", "fbclid"}

def canonical_url(u: str) -> str:
    try:
        p = urlparse(u)
        q = [(k, v) for (k, v) in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in _TRACKING_PARAMS]
        p2 = p._replace(query=urlencode(q, doseq=True), fragment="", netloc=p.netloc.lower())
        return urlunparse(p2)
    except: return u

def find_matches(hay: str, words: Iterable[str]) -> List[str]:
    hits = []
    for w in words or []:
        w2 = (w or "").strip()
        if not w2: continue
        if (" " in w2 and w2 in hay) or (" " not in w2 and re.search(rf"\b{re.escape(w2)}\b", hay)):
            hits.append(w2)
    return hits

def match_topic_with_reason(title: str, summary: str):
    hay = f"{title.lower()} {summary.lower()}"
    stop_hits = find_matches(hay, STOPWORDS)
    if stop_hits: return False, {"stopwords": stop_hits}
    if CFG.get("REQUIRE_KEYWORDS", True) and not KEYWORDS: return False, {"reason": "no_keywords"}
    
    narrow = [k for k in KEYWORDS if k not in BROAD_TOKENS]
    hits = find_matches(hay, narrow if narrow else KEYWORDS)
    return (True, {"keywords": hits, "narrow": bool(narrow)}) if hits else (False, {"keywords": []})

# ---------------------------------------------------------------------------
# Data Providers (NEW ARCHITECTURE)
# ---------------------------------------------------------------------------

class BaseProvider:
    """Базовый класс для всех источников данных"""
    def fetch(self) -> List[Tuple[Optional[int], bool, str, str, str]]:
        return []

class RSSProvider(BaseProvider):
    """Обычный парсинг RSS лент (твой старый функционал)"""
    def __init__(self, sources: List[str]):
        self.sources = sources

    def _parse_feed(self, url: str):
        try:
            d = feedparser.parse(url)
            items = []
            for e in d.entries:
                title = norm(getattr(e, "title", ""))
                link = norm(getattr(e, "link", ""))
                summary = norm(getattr(e, "summary", getattr(e, "description", "")))
                pp = getattr(e, "published_parsed", getattr(e, "updated_parsed", None))
                has_pub = pp is not None
                ts = int(time.mktime(pp)) if has_pub else None
                items.append((ts, has_pub, link, title, summary))
            return items
        except Exception as e:
            LOGGER.debug("RSS feed error %s: %s", url, e)
            return []

    def fetch(self):
        LOGGER.info(f"RSSProvider: Fetching {len(self.sources)} feeds...")
        all_items = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for items in ex.map(self._parse_feed, self.sources):
                all_items.extend(items)
        return all_items

class ArxivProvider(BaseProvider):
    """Научные статьи по IT, AI, криптографии"""
    def fetch(self):
        LOGGER.info("ArxivProvider: Fetching research papers...")
        # Собираем самые релевантные слова для науки (чтобы не превысить лимит URL)
        search_terms = ["osint", "phishing", "smishing", "cybersecurity", "geofencing", "data breach", "sim swap", "robocall"]
        items = []
        for term in search_terms:
            q = quote_plus(f"all:{term}")
            url = f"http://export.arxiv.org/api/query?search_query={q}&sortBy=submittedDate&sortOrder=desc&max_results=5"
            try:
                d = feedparser.parse(url)
                for e in d.entries:
                    title = norm(getattr(e, "title", ""))
                    link = norm(getattr(e, "link", ""))
                    summary = norm(getattr(e, "summary", ""))
                    pp = getattr(e, "published_parsed", None)
                    has_pub = pp is not None
                    ts = int(time.mktime(pp)) if has_pub else None
                    items.append((ts, has_pub, link, f"[Paper] {title}", summary))
            except Exception as e:
                LOGGER.debug("ArxivProvider error: %s", e)
        return items

class CrossrefProvider(BaseProvider):
    """База рецензируемых научных публикаций"""
    def fetch(self):
        LOGGER.info("CrossrefProvider: Fetching academic publications...")
        search_terms = ["open source intelligence", "phishing detection", "caller id spoofing"]
        items = []
        headers = {"User-Agent": "HeyLocate-Aggregator/1.0 (mailto:bot@example.com)"}
        for term in search_terms:
            url = f"https://api.crossref.org/works?query={quote_plus(term)}&select=title,URL,abstract,created&sort=created&order=desc&rows=5"
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if not r.ok: continue
                for item in r.json().get("message", {}).get("items", []):
                    title = item.get("title", [""])[0]
                    link = item.get("URL", "")
                    summary = norm(item.get("abstract", ""))
                    
                    # Пытаемся вытащить timestamp
                    created = item.get("created", {}).get("timestamp")
                    ts = int(created / 1000) if created else int(time.time())
                    items.append((ts, True, link, f"[Research] {title}", summary))
            except Exception as e:
                LOGGER.debug("CrossrefProvider error: %s", e)
        return items

class HackerNewsProvider(BaseProvider):
    """Тренды IT-дискуссий (Y Combinator)"""
    def fetch(self):
        LOGGER.info("HackerNewsProvider: Fetching latest tech discussions...")
        items = []
        try:
            url = "https://hacker-news.firebaseio.com/v0/newstories.json"
            story_ids = requests.get(url, timeout=10).json()[:50] # Берем 50 последних постов
            
            def fetch_hn_item(item_id):
                item_url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
                data = requests.get(item_url, timeout=5).json()
                if data and "title" in data and "url" in data:
                    return (data.get("time"), True, data["url"], f"[HN] {data['title']}", "")
                return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
                for res in ex.map(fetch_hn_item, story_ids):
                    if res: items.append(res)
        except Exception as e:
            LOGGER.debug("HackerNewsProvider error: %s", e)
        return items

class GitHubProvider(BaseProvider):
    """Поиск новых OSINT скриптов и инструментов на GitHub"""
    def fetch(self):
        LOGGER.info("GitHubProvider: Fetching trending repositories...")
        items = []
        headers = {"Accept": "application/vnd.github.v3+json"}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"token {GITHUB_TOKEN}"
        
        search_terms = ["osint", "phone-tracker", "caller-id"]
        for term in search_terms:
            # Ищем репозитории, обновленные за последние 2-3 дня
            url = f"https://api.github.com/search/repositories?q={term}+pushed:>2024-01-01&sort=updated&order=desc&per_page=5"
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if not r.ok: continue
                for repo in r.json().get("items", []):
                    title = repo.get("full_name", "")
                    link = repo.get("html_url", "")
                    summary = repo.get("description", "")
                    
                    # GitHub API возвращает ISO формат (2024-03-10T15:00:00Z)
                    updated = repo.get("updated_at")
                    try:
                        ts = int(datetime.strptime(updated, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
                    except:
                        ts = int(time.time())
                        
                    items.append((ts, True, link, f"[GitHub] {title}", summary))
            except Exception as e:
                LOGGER.debug("GitHubProvider error: %s", e)
        return items

# ---------------------------------------------------------------------------
# Core Aggregation Logic
# ---------------------------------------------------------------------------

def gather():
    """Собираем данные из всех провайдеров (RSS, Наука, Патенты, Соцсети)"""
    now = int(time.time())
    age_floor = now - AGE_LIMIT_SEC

    providers = [
        RSSProvider(SOURCES),
        ArxivProvider(),
        CrossrefProvider(),
        HackerNewsProvider(),
        GitHubProvider()
    ]

    all_raw_items = []
    # Запускаем провайдеров параллельно для максимальной скорости
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as ex:
        futures = {ex.submit(p.fetch): p.__class__.__name__ for p in providers}
        for future in concurrent.futures.as_completed(futures):
            p_name = futures[future]
            try:
                all_raw_items.extend(future.result())
            except Exception as e:
                LOGGER.error("Provider %s failed: %s", p_name, e)

    themed = []
    dropped_age = 0
    dropped_no_pub = 0

    for ts, has_pub, link, title, summary in all_raw_items:
        if CFG.get("REQUIRE_PUBDATE", True) and not has_pub:
            dropped_no_pub += 1
            continue
        if has_pub and ts is not None and ts < age_floor:
            dropped_age += 1
            continue

        ok, why = match_topic_with_reason(title, summary)
        if ok:
            ts_eff = ts if ts is not None else now
            themed.append((ts_eff, link, title, summary, why))

    themed.sort(key=lambda x: x[0], reverse=True)

    seen_run: Set[str] = set()
    unique = []
    for ts, link, title, summary, why in themed:
        if not link: continue
        link_c = canonical_url(link) if CFG.get("CANONICALIZE", True) else link
        domain = normalize_domain(urlparse(link_c).netloc)
        
        if domain in BLOCKED_DOMAINS: continue
        if link_c not in seen_run:
            seen_run.add(link_c)
            unique.append((ts, link_c, title, summary, domain, why))
        if len(unique) >= CFG["LIMIT_PER_RUN"]:
            break

    LOGGER.info("[debug] Total kept=%d (dropped_age=%d, dropped_no_pub=%d)", len(unique), dropped_age, dropped_no_pub)
    return unique

# ---------------------------------------------------------------------------
# Seen Store, Fetcher & Semantic Rerank (Оставлено без изменений)
# ---------------------------------------------------------------------------

def load_seen() -> Dict[str, str]:
    if not SEEN_PATH.exists(): return {}
    try:
        data = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            now = datetime.now(timezone.utc).isoformat()
            return {u: now for u in data}
        return data if isinstance(data, dict) else {}
    except: return {}

def save_seen(seen: Dict[str, str]):
    atomic_write_text(SEEN_PATH, json.dumps(seen, ensure_ascii=False, indent=2))

def prune_seen(seen: Dict[str, str], keep: int = 50000) -> Dict[str, str]:
    if len(seen) <= keep: return seen
    def ts_or_min(iso: str) -> datetime:
        try: return datetime.fromisoformat(iso.replace("Z", ""))
        except: return datetime.min.replace(tzinfo=timezone.utc)
    return dict(sorted(seen.items(), key=lambda kv: ts_or_min(kv[1]), reverse=True)[:keep])

def any_word_simple(hay: str, words: Iterable[str]) -> bool:
    return any((w.strip().lower() in hay) for w in words or [] if w.strip())

def categorize_item(title: str, summary: str, domain: str) -> str:
    hay = f"{title.lower()} {summary.lower()}"
    for cat in CATEGORIES:
        name = cat.get("name") or CFG["UNCATEGORIZED"]
        inc_any = [s.lower() for s in (cat.get("include_any") or [])]
        inc_all = [s.lower() for s in (cat.get("include_all") or [])]
        exc_any = [s.lower() for s in (cat.get("exclude_any") or [])]
        doms = [normalize_domain(s) for s in (cat.get("domains") or [])]

        if doms and domain in doms:
            if exc_any and any_word_simple(hay, exc_any): continue
            return name
        if inc_any and not any_word_simple(hay, inc_any): continue
        if inc_all and not all(any_word_simple(hay, [w]) for w in inc_all): continue
        if exc_any and any_word_simple(hay, exc_any): continue
        return name
    return CFG["UNCATEGORIZED"]

def _fetch_text(url: str) -> str:
    if trafilatura is not None:
        try:
            html = trafilatura.fetch_url(url, timeout=12)
            if html:
                txt = trafilatura.extract(html, include_comments=False) or ""
                if txt: return txt.strip()
        except: pass
    if requests is not None:
        try:
            r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True)
            if r.ok and r.text:
                if trafilatura is not None:
                    txt = trafilatura.extract(r.text, include_comments=False) or ""
                    return (txt or "").strip()
                return r.text[:5000]
        except: pass
    return ""

def fetch_text_cached(url: str) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    p = CACHE_DIR / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".txt")
    if p.exists():
        try: return p.read_text(encoding="utf-8", errors="ignore")
        except: pass
    txt = _fetch_text(url)
    try: p.write_text(txt, encoding="utf-8")
    except: pass
    return txt

_MODEL = None
_CAT_QUERIES_ENCODED = None

def _get_model():
    global _MODEL
    if _MODEL is not None: return _MODEL
    try:
        from FlagEmbedding import BGEM3FlagModel
        LOGGER.info("Loading BGE-M3 model...")
        _MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
        return _MODEL
    except Exception as e:
        LOGGER.error("Failed to load BGE-M3 model: %s", e)
        raise

def _chunks(s: str, n: int) -> Iterable[str]:
    for i in range(0, len(s), n): yield s[i : i + n]

def _build_query_for_category(cat: dict) -> str:
    inc_any = cat.get("include_any") or []
    inc_all = cat.get("include_all") or []
    parts = (inc_all[:3] if inc_all else []) + (inc_any[:5] if inc_any else [])
    return " ; ".join(parts) or "mobile security telecom lookup"

def log_reason(payload: dict):
    if not CFG.get("LOG_REASONS", True): return
    try:
        with open(REASONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except: pass

def _rule_only_grouping(items):
    grouped = {c.get("name") or CFG["UNCATEGORIZED"]: [] for c in CATEGORIES}
    grouped[CFG["UNCATEGORIZED"]] = []
    score_map = {}
    for ts, link, title, summary, domain, why in items:
        cat = categorize_item(title, summary, domain)
        grouped.setdefault(cat, []).append(link)
        log_reason({"ts": datetime.now(timezone.utc).isoformat(), "link": link, "domain": domain, "category": cat, "semantic": False})
    return grouped, score_map

def semantic_filter_and_rank(items):
    if not SEMANTIC_RERANK or trafilatura is None:
        return _rule_only_grouping(items)

    try: model = _get_model()
    except: return _rule_only_grouping(items)

    cat_queries = [(c.get("name") or CFG["UNCATEGORIZED"], _build_query_for_category(c)) for c in CATEGORIES]
    cat_queries.append((CFG["UNCATEGORIZED"], "mobile security telecom lookup"))
    
    global _CAT_QUERIES_ENCODED
    if _CAT_QUERIES_ENCODED is None:
        _CAT_QUERIES_ENCODED = model.encode([q for _, q in cat_queries], return_dense=True, return_sparse=True, return_colbert_vecs=True)
    q_out = _CAT_QUERIES_ENCODED

    to_score = items[:CFG["MAX_FETCH"]]
    urls = [link for _, link, *_ in to_score]
    texts = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(CFG.get("MAX_WORKERS", 8))) as ex:
        for url, txt in zip(urls, ex.map(fetch_text_cached, urls)):
            texts[url] = txt

    kept = {name: [] for name, _ in cat_queries}
    score_map = {name: {} for name, _ in cat_queries}

    for (ts, link, title, summary, domain, why) in to_score:
        text = texts.get(link) or ""
        if not text: continue
        try:
            pieces = list(_chunks(text[:CFG["MAX_TXT_LEN"]], CFG["CHUNK_SIZE"]))
            if not pieces: continue
            d_out = model.encode(pieces, return_dense=True, return_sparse=True, return_colbert_vecs=True)
            best_scores = [0.0] * len(cat_queries)
            for p_idx in range(len(pieces)):
                for c_idx in range(len(cat_queries)):
                    d = float((q_out["dense_vecs"][c_idx] @ d_out["dense_vecs"][p_idx].T)[0, 0])
                    s = float(model.compute_lexical_matching_score(q_out["lexical_weights"][c_idx], d_out["lexical_weights"][p_idx]))
                    c = float(model.colbert_score(q_out["colbert_vecs"][c_idx], d_out["colbert_vecs"][p_idx]))
                    h = CFG["MV_WEIGHTS"][0]*d + CFG["MV_WEIGHTS"][1]*s + CFG["MV_WEIGHTS"][2]*c
                    if h > best_scores[c_idx]: best_scores[c_idx] = h
            
            for idx, (name, q_str) in enumerate(cat_queries):
                if best_scores[idx] >= CFG["THRESHOLD"]:
                    kept[name].append((best_scores[idx], link))
                    score_map.setdefault(name, {})[link] = best_scores[idx]
                    log_reason({"ts": datetime.now(timezone.utc).isoformat(), "link": link, "category": name, "score": round(best_scores[idx], 4), "semantic": True})
                    break
        except: pass

    grouped = {name: [u for s, u in sorted(vals, reverse=True)[:CFG["TOPK_PER_CAT"]]] for name, vals in kept.items()}
    if sum(len(v) for v in grouped.values()) == 0 and CFG.get("FALLBACK_IF_EMPTY", True):
        return _rule_only_grouping(items)
    return grouped, score_map

def write_outputs(grouped, full_by_cat):
    atomic_write_text(LINKS_JSON, json.dumps(grouped, ensure_ascii=False, indent=2))
    atomic_write_text(LINKS_FULL_JSON, json.dumps(full_by_cat, ensure_ascii=False, indent=2))
    lines = []
    for cat, urls in grouped.items():
        if urls: lines.extend([f"## {cat}"] + urls + [""])
    atomic_write_text(LINKS_TXT, ("\n".join(lines).strip() + "\n") if lines else "")

def main():
    LOGGER.info("Starting news-links aggregation run")
    CACHE_DIR.mkdir(exist_ok=True)

    items = gather()
    seen = load_seen()
    now_iso = datetime.now(timezone.utc).isoformat()

    new_items = [it for it in items if it[1] not in seen]
    grouped, score_map = semantic_filter_and_rank(new_items)

    full_by_cat = {}
    items_by_url = {link: (ts, title, summary, domain, why) for ts, link, title, summary, domain, why in new_items}
    for cat, urls in grouped.items():
        cat_records = []
        for url in urls:
            ts, title, summary, domain, why = items_by_url.get(url, (None, "", "", "", {}))
            cat_records.append({"url": url, "title": title, "summary": summary, "domain": domain, "score": score_map.get(cat, {}).get(url)})
        full_by_cat[cat] = cat_records

    write_outputs(grouped, full_by_cat)

    new_links = {u for urls in grouped.values() for u in urls}
    for link in new_links: seen[link] = now_iso
    save_seen(prune_seen(seen))
    
    with open(ARCHIVE_TXT, "a", encoding="utf-8") as f:
        for link in sorted(new_links): f.write(link + "\n")

    LOGGER.info("[OK] Total categorized links saved: %d", len(new_links))

if __name__ == "__main__":
    main()
