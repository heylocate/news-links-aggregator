import os
import json
import time
import re
import pathlib
import hashlib
import logging
import asyncio
import sqlite3
import pickle
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable, Set, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, quote_plus

import aiohttp
import feedparser

try:
    import trafilatura
except Exception:
    trafilatura = None

ROOT = pathlib.Path(__file__).parent

DB_PATH = ROOT / "heylocate.db"
SEEN_PATH_OLD = ROOT / "seen.json"
LINKS_TXT = ROOT / "links.txt"
LINKS_JSON = ROOT / "links_by_category.json"
LINKS_FULL_JSON = ROOT / "links_by_category_full.json"
ARCHIVE_TXT = ROOT / "archive_links.txt"
CFG_PATH = ROOT / "muvera_config.json"
CAT_PATH = ROOT / "categories.json"

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
# SQLite Database & Migration
# ---------------------------------------------------------------------------

def init_db():
    """Инициализация таблиц и авто-миграция со старого формата JSON"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS seen_urls (url TEXT PRIMARY KEY, added_at TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS text_cache (url TEXT PRIMARY KEY, content TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS vector_cache (url TEXT PRIMARY KEY, dense_vec BLOB)")
        
        # Миграция старого seen.json
        if SEEN_PATH_OLD.exists():
            LOGGER.info("Found seen.json. Migrating data to SQLite...")
            try:
                old_seen = json.loads(SEEN_PATH_OLD.read_text(encoding="utf-8"))
                if isinstance(old_seen, dict):
                    conn.executemany("INSERT OR IGNORE INTO seen_urls (url, added_at) VALUES (?, ?)", old_seen.items())
                elif isinstance(old_seen, list):
                    now = datetime.now(timezone.utc).isoformat()
                    conn.executemany("INSERT OR IGNORE INTO seen_urls (url, added_at) VALUES (?, ?)", [(u, now) for u in old_seen])
                SEEN_PATH_OLD.rename(SEEN_PATH_OLD.with_suffix(".json.bak"))
                LOGGER.info("Migration successful. Renamed to seen.json.bak")
            except Exception as e:
                LOGGER.error("Migration failed: %s", e)

def is_seen(url: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("SELECT 1 FROM seen_urls WHERE url = ?", (url,)).fetchone() is not None

def mark_seen(urls: Iterable[str]):
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany("INSERT OR IGNORE INTO seen_urls (url, added_at) VALUES (?, ?)", [(u, now) for u in urls])
        # Авто-очистка: храним только последние 50,000 ссылок
        count = conn.execute("SELECT COUNT(*) FROM seen_urls").fetchone()[0]
        if count > 50000:
            conn.execute("DELETE FROM seen_urls WHERE url NOT IN (SELECT url FROM seen_urls ORDER BY added_at DESC LIMIT 50000)")

def log_reason(payload: dict):
    if not CFG.get("LOG_REASONS", True): return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO logs (payload) VALUES (?)", (json.dumps(payload, ensure_ascii=False),))

# ---------------------------------------------------------------------------
# Config & Keyword Matching
# ---------------------------------------------------------------------------

CFG = {"REQUIRE_KEYWORDS": True, "LIMIT_PER_RUN": 800, "MAX_FETCH": 30, "MAX_TXT_LEN": 4000, "CHUNK_SIZE": 700, "TOPK_PER_CAT": 12, "MV_WEIGHTS": [0.2, 0.2, 0.6], "THRESHOLD": 0.50, "UNCATEGORIZED": "Uncategorized"}
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
# ASYNC Data Providers
# ---------------------------------------------------------------------------

class BaseProvider:
    async def fetch(self, session: aiohttp.ClientSession) -> List[Tuple[Optional[int], bool, str, str, str]]:
        return []

class RSSProvider(BaseProvider):
    def __init__(self, sources: List[str]):
        self.sources = sources

    async def fetch(self, session):
        LOGGER.info(f"RSSProvider: Fetching {len(self.sources)} feeds asynchronously...")
        all_items = []
        
        async def fetch_one(url):
            try:
                async with session.get(url, timeout=10) as resp:
                    text = await resp.text()
                    d = await asyncio.to_thread(feedparser.parse, text)
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
                return []

        tasks = [fetch_one(u) for u in self.sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, list): all_items.extend(res)
        return all_items

class ArxivProvider(BaseProvider):
    async def fetch(self, session):
        LOGGER.info("ArxivProvider: Fetching research papers...")
        search_terms = ["osint", "phishing", "smishing", "cybersecurity", "geofencing"]
        items = []
        
        async def fetch_term(term):
            q = quote_plus(f"all:{term}")
            url = f"http://export.arxiv.org/api/query?search_query={q}&sortBy=submittedDate&sortOrder=desc&max_results=5"
            try:
                async with session.get(url, timeout=10) as resp:
                    d = await asyncio.to_thread(feedparser.parse, await resp.text())
                    return [(int(time.mktime(e.published_parsed)) if getattr(e, "published_parsed", None) else None, 
                             True, norm(getattr(e, "link", "")), f"[Paper] {norm(getattr(e, 'title', ''))}", 
                             norm(getattr(e, "summary", ""))) for e in d.entries]
            except: return []

        results = await asyncio.gather(*[fetch_term(t) for t in search_terms])
        for res in results: items.extend(res)
        return items

class HackerNewsProvider(BaseProvider):
    async def fetch(self, session):
        LOGGER.info("HackerNewsProvider: Fetching latest tech discussions...")
        items = []
        try:
            url = "https://hacker-news.firebaseio.com/v0/newstories.json"
            async with session.get(url, timeout=10) as resp:
                story_ids = (await resp.json())[:50]
            
            async def fetch_hn_item(item_id):
                item_url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
                try:
                    async with session.get(item_url, timeout=5) as r:
                        data = await r.json()
                        if data and "title" in data and "url" in data:
                            return (data.get("time"), True, data["url"], f"[HN] {data['title']}", "")
                except: pass
                return None

            results = await asyncio.gather(*[fetch_hn_item(i) for i in story_ids])
            items = [res for res in results if res]
        except Exception as e:
            LOGGER.debug("HackerNewsProvider error: %s", e)
        return items

class GitHubProvider(BaseProvider):
    async def fetch(self, session):
        LOGGER.info("GitHubProvider: Fetching trending repositories...")
        items = []
        headers = {"Accept": "application/vnd.github.v3+json"}
        if GITHUB_TOKEN: headers["Authorization"] = f"token {GITHUB_TOKEN}"
        
        search_terms = ["osint", "phone-tracker", "caller-id"]
        
        async def fetch_gh(term):
            url = f"https://api.github.com/search/repositories?q={term}+pushed:>2024-01-01&sort=updated&order=desc&per_page=5"
            try:
                async with session.get(url, headers=headers, timeout=10) as r:
                    if not r.ok: return []
                    data = await r.json()
                    local_items = []
                    for repo in data.get("items", []):
                        updated = repo.get("updated_at")
                        try: ts = int(datetime.strptime(updated, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
                        except: ts = int(time.time())
                        local_items.append((ts, True, repo.get("html_url", ""), f"[GitHub] {repo.get('full_name', '')}", repo.get("description", "")))
                    return local_items
            except: return []

        results = await asyncio.gather(*[fetch_gh(t) for t in search_terms])
        for res in results: items.extend(res)
        return items

# ---------------------------------------------------------------------------
# ASYNC Core Aggregation Logic
# ---------------------------------------------------------------------------

async def gather_async(session):
    now = int(time.time())
    age_floor = now - AGE_LIMIT_SEC

    providers = [RSSProvider(SOURCES), ArxivProvider(), HackerNewsProvider(), GitHubProvider()]
    
    all_raw_items = []
    results = await asyncio.gather(*[p.fetch(session) for p in providers], return_exceptions=True)
    for res in results:
        if isinstance(res, list): all_raw_items.extend(res)

    themed = []
    dropped_age = dropped_no_pub = 0

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
        
        if domain in BLOCKED_DOMAINS or is_seen(link_c): continue
        if link_c not in seen_run:
            seen_run.add(link_c)
            unique.append((ts, link_c, title, summary, domain, why))
        if len(unique) >= CFG["LIMIT_PER_RUN"]: break

    LOGGER.info("[debug] Total kept=%d (dropped_age=%d, dropped_no_pub=%d)", len(unique), dropped_age, dropped_no_pub)
    return unique

# ---------------------------------------------------------------------------
# ASYNC Text Fetching (with DB cache)
# ---------------------------------------------------------------------------

async def fetch_text_async(url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT content FROM text_cache WHERE url = ?", (url,)).fetchone()
        if row: return row[0]

    async with semaphore:
        try:
            async with session.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True) as resp:
                html = await resp.text()
                if trafilatura is not None:
                    txt = await asyncio.to_thread(trafilatura.extract, html, include_comments=False)
                    txt = (txt or "").strip()
                else: txt = html[:5000]
                
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute("INSERT OR REPLACE INTO text_cache (url, content) VALUES (?, ?)", (url, txt))
                return txt
        except Exception:
            return ""

# ---------------------------------------------------------------------------
# SYNC Semantic Rerank (with Vector DB Cache)
# ---------------------------------------------------------------------------

_MODEL = None
_CAT_QUERIES_ENCODED = None

def _get_model():
    global _MODEL
    if _MODEL is not None: return _MODEL
    from FlagEmbedding import BGEM3FlagModel
    LOGGER.info("Loading BGE-M3 model...")
    _MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    return _MODEL

def _chunks(s: str, n: int) -> Iterable[str]:
    for i in range(0, len(s), n): yield s[i : i + n]

def _build_query_for_category(cat: dict) -> str:
    inc_any = cat.get("include_any") or []
    inc_all = cat.get("include_all") or []
    parts = (inc_all[:3] if inc_all else []) + (inc_any[:5] if inc_any else [])
    return " ; ".join(parts) or "mobile security telecom lookup"

def _rule_only_grouping(items):
    grouped = {c.get("name") or CFG["UNCATEGORIZED"]: [] for c in CATEGORIES}
    grouped[CFG["UNCATEGORIZED"]] = []
    score_map = {}
    for ts, link, title, summary, domain, why in items:
        cat = categorize_item(title, summary, domain)
        grouped.setdefault(cat, []).append(link)
        log_reason({"ts": datetime.now(timezone.utc).isoformat(), "link": link, "domain": domain, "category": cat, "semantic": False})
    return grouped, score_map

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

def semantic_filter_and_rank(items, texts):
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

    kept = {name: [] for name, _ in cat_queries}
    score_map = {name: {} for name, _ in cat_queries}

    for (ts, link, title, summary, domain, why) in items:
        text = texts.get(link) or ""
        if not text: continue
        
        try:
            # 1. Пытаемся достать векторы из SQLite
            with sqlite3.connect(DB_PATH) as conn:
                row = conn.execute("SELECT dense_vec FROM vector_cache WHERE url = ?", (link,)).fetchone()
                
            if row:
                d_out = pickle.loads(row[0]) # Мгновенная загрузка!
            else:
                pieces = list(_chunks(text[:CFG["MAX_TXT_LEN"]], CFG["CHUNK_SIZE"]))
                if not pieces: continue
                # Тяжелая операция CPU (работает только для новых текстов)
                d_out = model.encode(pieces, return_dense=True, return_sparse=True, return_colbert_vecs=True)
                
                # Сохраняем результат в SQLite для будущих запусков
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute("INSERT OR REPLACE INTO vector_cache (url, dense_vec) VALUES (?, ?)", (link, sqlite3.Binary(pickle.dumps(d_out))))
            
            best_scores = [0.0] * len(cat_queries)
            for p_idx in range(len(d_out["dense_vecs"])):
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
        except Exception as e:
            LOGGER.error(f"Semantic error for {link}: {e}")

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

async def main_async():
    LOGGER.info("Starting Async news-links aggregation run")
    init_db()

    async with aiohttp.ClientSession() as session:
        # 1. Асинхронный сбор всех ссылок со всех сайтов и API (за секунды)
        items = await gather_async(session)
        
        # 2. Асинхронное скачивание текстов статей (ограничиваем до 20 одновременных загрузок, чтобы не забанили)
        to_score = items[:CFG["MAX_FETCH"]]
        urls = [link for _, link, *_ in to_score]
        semaphore = asyncio.Semaphore(20)
        
        LOGGER.info(f"Fetching {len(urls)} texts asynchronously...")
        fetch_tasks = [fetch_text_async(u, session, semaphore) for u in urls]
        fetched_texts = await asyncio.gather(*fetch_tasks)
        texts = dict(zip(urls, fetched_texts))

    # 3. Синхронное семантическое вычисление (на CPU)
    grouped, score_map = semantic_filter_and_rank(to_score, texts)

    full_by_cat = {}
    items_by_url = {link: (ts, title, summary, domain, why) for ts, link, title, summary, domain, why in to_score}
    for cat, urls in grouped.items():
        cat_records = []
        for url in urls:
            ts, title, summary, domain, why = items_by_url.get(url, (None, "", "", "", {}))
            cat_records.append({"url": url, "title": title, "summary": summary, "domain": domain, "score": score_map.get(cat, {}).get(url)})
        full_by_cat[cat] = cat_records

    # 4. Сохранение данных и запись в БД
    write_outputs(grouped, full_by_cat)

    new_links = {u for urls in grouped.values() for u in urls}
    mark_seen(new_links)
    
    with open(ARCHIVE_TXT, "a", encoding="utf-8") as f:
        for link in sorted(new_links): f.write(link + "\n")

    LOGGER.info("[OK] Total categorized links saved: %d", len(new_links))

if __name__ == "__main__":
    # Запускаем асинхронный цикл событий
    asyncio.run(main_async())
