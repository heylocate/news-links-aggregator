import os
import json
import time
import re
import pathlib
import hashlib
import logging
import concurrent.futures
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Iterable, Set
from urllib.parse import (
    urlparse,
    urlunparse,
    parse_qsl,
    urlencode,
    urljoin,
    quote_plus,
)

import feedparser  # type: ignore

# optional heavy deps
try:
    import trafilatura  # type: ignore
except Exception:  # pragma: no cover
    trafilatura = None
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

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

# allow toggle from workflow via env
SEMANTIC_RERANK = os.getenv("HL_SEMANTIC_RERANK", "true").lower() in ("1", "true", "yes", "y")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("heylocate.news_aggregator")
if not LOGGER.handlers:
    log_level_name = os.getenv("HL_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    LOGGER.setLevel(log_level)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_lines(path: pathlib.Path) -> List[str]:
    if not path.exists():
        return []
    return [
        l.strip()
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]


def load_json(path: pathlib.Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        LOGGER.warning("Failed to load JSON from %s: %s", path, e)
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


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    """Атомарная запись: пишем во временный файл и заменяем основной."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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
    "MAX_AGE_DAYS": 1,  # only fresh items (≤ 1 day)
    "REQUIRE_PUBDATE": True,  # drop entries without pubdate
    "DISCOVERY": {
        "enabled": True,
        "category_name": "Outside Sources (Top 10)",
        "max_links": 10,
        "per_query_limit": 30,
        "recency": "when:1d",
        # language-only mode (no geo)
        "language_only": True,
        "lang": "en",
        # domain harvesting
        "save_domains": True,
        "save_domains_file": "discovered_sources.txt",
        # fallback CEID
        "fallback_ceid_if_empty": True,
        "compat_ceids": ["US:en", "GB:en"],
    },
}

CFG.update(load_json(CFG_PATH, {}))
AGE_LIMIT_SEC = int(CFG.get("MAX_AGE_DAYS", 1)) * 86400

SOURCES = read_lines(ROOT / "sources.txt")
KEYWORDS = [s.lower() for s in read_lines(ROOT / "keywords.txt")]
STOPWORDS = [s.lower() for s in read_lines(ROOT / "stopwords.txt")]
BLOCKED_DOMAINS = {normalize_domain(d) for d in read_lines(ROOT / "blocked_domains.txt")}
CATEGORIES = load_json(CAT_PATH, [])

BROAD_TOKENS = {
    "android", "iphone", "ios", "ipados", "watchos", "wear os",
    "5g", "lte", "volte", "vowifi",
    "samsung", "galaxy", "pixel", "oneplus", "xiaomi", "oppo", "vivo", "nokia", "motorola",
    "phone", "smartphone", "mobile", "cellular",
}

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id",
    "utm_name", "utm_cid", "utm_reader", "gclid", "fbclid", "mc_cid", "mc_eid", "igshid",
}


# ---------------------------------------------------------------------------
# URL canonicalization
# ---------------------------------------------------------------------------

def canonical_url(u: str) -> str:
    try:
        p = urlparse(u)
        q = [
            (k, v)
            for (k, v) in parse_qsl(p.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_PARAMS
        ]
        p2 = p._replace(
            query=urlencode(q, doseq=True),
            fragment="",
            netloc=p.netloc.lower(),
        )
        u2 = urlunparse(p2)
        if trafilatura is not None and CFG.get("CANONICALIZE", True):
            html = trafilatura.fetch_url(u2, timeout=8)
            if html:
                m = re.search(
                    r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']',
                    html,
                    re.I,
                )
                if m:
                    href = m.group(1)
                    if not urlparse(href).netloc:
                        href = urljoin(u2, href)
                    return href
        return u2
    except Exception as e:  # pragma: no cover
        LOGGER.debug("canonical_url error for %s: %s", u, e)
        return u


# ---------------------------------------------------------------------------
# Keyword / stopword matching
# ---------------------------------------------------------------------------

def find_matches(hay: str, words: Iterable[str]) -> List[str]:
    hits: List[str] = []
    for w in words or []:
        w2 = (w or "").strip()
        if not w2:
            continue
        if (" " in w2 and w2 in hay) or (
            " " not in w2 and re.search(rf"\b{re.escape(w2)}\b", hay)
        ):
            hits.append(w2)
    return hits


def match_topic_with_reason(title: str, summary: str):
    hay = f"{title.lower()} {summary.lower()}"
    stop_hits = find_matches(hay, STOPWORDS)
    if stop_hits:
        return False, {"stopwords": stop_hits}
    if CFG.get("REQUIRE_KEYWORDS", True) and not KEYWORDS:
        return False, {"reason": "no_keywords"}
    narrow = [k for k in KEYWORDS if k not in BROAD_TOKENS]
    if narrow:
        hits = find_matches(hay, narrow)
        return (True, {"keywords": hits, "narrow": True}) if hits else (
            False,
            {"keywords": [], "narrow": True},
        )
    else:
        hits = find_matches(hay, KEYWORDS)
        return (True, {"keywords": hits, "narrow": False}) if hits else (
            False,
            {"keywords": []},
        )


# ---------------------------------------------------------------------------
# Feed parsing / gathering
# ---------------------------------------------------------------------------

def parse_feed(url: str):
    d = feedparser.parse(url)
    items = []
    for e in d.entries:
        title = norm(getattr(e, "title", ""))
        link = norm(getattr(e, "link", ""))
        summary = norm(getattr(e, "summary", getattr(e, "description", "")))
        pp = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
        has_pub = pp is not None
        ts = int(time.mktime(pp)) if has_pub else None
        items.append((ts, has_pub, link, title, summary))
    return items


def _parse_feed_safe(url: str):
    try:
        return parse_feed(url)
    except Exception as e:  # pragma: no cover
        LOGGER.warning("Feed error: %s (%s)", url, e)
        return []


def gather():
    """Собираем все свежие уникальные ссылки из RSS."""
    now = int(time.time())
    age_floor = now - AGE_LIMIT_SEC

    all_items = []
    if not SOURCES:
        LOGGER.warning("No sources defined in sources.txt")
        return []

    max_workers = int(CFG.get("MAX_WORKERS", 8))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for items in ex.map(_parse_feed_safe, SOURCES):
            all_items.extend(items)

    themed = []
    dropped_age = 0
    dropped_no_pub = 0

    for ts, has_pub, link, title, summary in all_items:
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
        LOGGER.info(
            "[debug] kept=%d (dropped_age=%d, dropped_no_pub=%d) age_floor=%s",
            len(unique),
            dropped_age,
            dropped_no_pub,
            dt.fromtimestamp(age_floor, tz=timezone.utc).isoformat(),
        )

    return unique


# ---------------------------------------------------------------------------
# Seen store
# ---------------------------------------------------------------------------

def load_seen() -> Dict[str, str]:
    if not SEEN_PATH.exists():
        return {}
    try:
        data = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            now = datetime.now(timezone.utc).isoformat()
            return {u: now for u in data}
        return data if isinstance(data, dict) else {}
    except Exception as e:  # pragma: no cover
        LOGGER.warning("Failed to load seen.json: %s", e)
        return {}


def save_seen(seen: Dict[str, str]):
    text = json.dumps(seen, ensure_ascii=False, indent=2)
    atomic_write_text(SEEN_PATH, text)


def prune_seen(seen: Dict[str, str], keep: int = 50000) -> Dict[str, str]:
    if len(seen) <= keep:
        return seen

    def ts_or_min(iso: str) -> datetime:
        try:
            return datetime.fromisoformat(iso.replace("Z", ""))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    items = sorted(seen.items(), key=lambda kv: ts_or_min(kv[1]), reverse=True)[:keep]
    return dict(items)


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------

def any_word_simple(hay: str, words: Iterable[str]) -> bool:
    for w in words or []:
        w2 = (w or "").strip().lower()
        if not w2:
            continue
        if w2 in hay:
            return True
    return False


def categorize_item(title: str, summary: str, domain: str) -> str:
    hay = f"{title.lower()} {summary.lower()}"
    for cat in CATEGORIES:
        name = cat.get("name") or CFG["UNCATEGORIZED"]
        inc_any = [s.lower() for s in (cat.get("include_any") or [])]
        inc_all = [s.lower() for s in (cat.get("include_all") or [])]
        exc_any = [s.lower() for s in (cat.get("exclude_any") or [])]
        doms = [normalize_domain(s) for s in (cat.get("domains") or [])]

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


# ---------------------------------------------------------------------------
# Robust text fetch
# ---------------------------------------------------------------------------

def _fetch_text(url: str) -> str:
    if trafilatura is not None:
        try:
            html = trafilatura.fetch_url(url, timeout=12)
            if html:
                txt = trafilatura.extract(html, include_comments=False) or ""
                if txt:
                    return txt.strip()
        except Exception as e:  # pragma: no cover
            LOGGER.debug("trafilatura.fetch_url error for %s: %s", url, e)
    if requests is not None:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (HL-NewsBot)"}
            r = requests.get(url, timeout=12, headers=headers, allow_redirects=True)
            if r.ok and r.text:
                if trafilatura is not None:
                    txt = trafilatura.extract(r.text, include_comments=False) or ""
                    return (txt or "").strip()
                return r.text[:5000]
        except Exception as e:  # pragma: no cover
            LOGGER.debug("requests.get error for %s: %s", url, e)
    return ""


def fetch_text_cached(url: str) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    p = CACHE_DIR / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".txt")
    if p.exists():
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:  # pragma: no cover
            LOGGER.debug("Failed to read cache %s: %s", p, e)
    txt = _fetch_text(url)
    try:
        p.write_text(txt, encoding="utf-8")
    except Exception as e:  # pragma: no cover
        LOGGER.debug("Failed to write cache %s: %s", p, e)
    return txt


# ---------------------------------------------------------------------------
# Semantic scoring (BGE-M3) with graceful fallback
# ---------------------------------------------------------------------------

_MODEL = None
_CAT_QUERIES_ENCODED = None  # cache for category query embeddings


def _get_model():
    """Lazy load BGE-M3 model with error handling."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore
        LOGGER.info("Loading BGE-M3 model for semantic rerank...")
        _MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
        return _MODEL
    except Exception as e:
        # Здесь как раз ловятся проблемы с HuggingFace cache / сетью
        LOGGER.error("Failed to load BGE-M3 model: %s", e)
        raise


def _chunks(s: str, n: int) -> Iterable[str]:
    for i in range(0, len(s), n):
        yield s[i : i + n]


def _build_query_for_category(cat: dict) -> str:
    inc_any = cat.get("include_any") or []
    inc_all = cat.get("include_all") or []
    parts = (inc_all[:3] if inc_all else []) + (inc_any[:5] if inc_any else [])
    return " ; ".join(parts) or "mobile security telecom lookup"


def _build_category_queries_with_names() -> List[Tuple[str, str]]:
    cat_queries = [
        (c.get("name") or CFG["UNCATEGORIZED"], _build_query_for_category(c))
        for c in CATEGORIES
    ]
    cat_queries.append((CFG["UNCATEGORIZED"], "mobile security telecom lookup"))
    return cat_queries


def _ensure_cat_queries_encoded(model, cat_queries: List[Tuple[str, str]]):
    global _CAT_QUERIES_ENCODED
    if _CAT_QUERIES_ENCODED is not None:
        return _CAT_QUERIES_ENCODED
    queries = [q for _, q in cat_queries]
    LOGGER.info("Encoding %d category queries for semantic scoring...", len(queries))
    q_out = model.encode(
        queries,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    _CAT_QUERIES_ENCODED = q_out
    return q_out


def mv_scores_for_all_categories(model, q_out, doc_text: str, mv_weights, max_txt: int, chunk_size: int) -> List[float]:
    text = doc_text[:max_txt]
    if not text.strip():
        return [0.0] * len(q_out["dense_vecs"])
    pieces = list(_chunks(text, chunk_size))
    if not pieces:
        return [0.0] * len(q_out["dense_vecs"])

    d_out = model.encode(
        pieces,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    q_dense = q_out["dense_vecs"]
    q_lex = q_out["lexical_weights"]
    q_colb = q_out["colbert_vecs"]

    d_dense = d_out["dense_vecs"]
    d_lex = d_out["lexical_weights"]
    d_colb = d_out["colbert_vecs"]

    n_cats = len(q_dense)
    best_scores = [0.0] * n_cats

    for piece_idx in range(len(pieces)):
        d_vec = d_dense[piece_idx]
        d_lex_w = d_lex[piece_idx]
        d_colb_vecs = d_colb[piece_idx]
        for cat_idx in range(n_cats):
            try:
                d = float((q_dense[cat_idx] @ d_vec.T)[0, 0])
                s = float(model.compute_lexical_matching_score(q_lex[cat_idx], d_lex_w))
                c = float(model.colbert_score(q_colb[cat_idx], d_colb_vecs))
                hybrid = mv_weights[0] * d + mv_weights[1] * s + mv_weights[2] * c
                if hybrid > best_scores[cat_idx]:
                    best_scores[cat_idx] = hybrid
            except Exception:
                continue

    return best_scores


def log_reason(payload: dict):
    if not CFG.get("LOG_REASONS", True):
        return
    try:
        with open(REASONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:  # pragma: no cover
        LOGGER.debug("Failed to write reasons.jsonl: %s", e)


def _rule_only_grouping(items):
    grouped: Dict[str, List[str]] = {}
    for cat in [c.get("name") for c in CATEGORIES] + [CFG["UNCATEGORIZED"]]:
        if cat:
            grouped[cat] = []
    score_map: Dict[str, Dict[str, float]] = {}

    for ts, link, title, summary, domain, why in items:
        cat = categorize_item(title, summary, domain)
        grouped.setdefault(cat, []).append(link)
        log_reason(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "link": link,
                "domain": domain,
                "category": cat,
                "semantic": False,
                "keyword_hits": (why.get("keywords") if isinstance(why, dict) else None),
            }
        )
    return grouped, score_map


def semantic_filter_and_rank(items):
    """
    items: [(ts, link, title, summary, domain, why)]
    Возвращает:
        grouped_urls: {category -> [url, ...]}
        score_map: {category -> {url -> score}}
    """
    if not SEMANTIC_RERANK or trafilatura is None:
        LOGGER.info("Semantic rerank disabled or trafilatura missing -> rule-only grouping")
        return _rule_only_grouping(items)

    mv_weights = CFG["MV_WEIGHTS"]
    threshold = CFG["THRESHOLD"]
    max_fetch = CFG["MAX_FETCH"]
    max_txt = CFG["MAX_TXT_LEN"]
    chunk_size = CFG["CHUNK_SIZE"]
    max_workers = int(CFG.get("MAX_WORKERS", 8))

    # Пытаемся загрузить модель. Если не получилось – падаем в rule-only.
    try:
        model = _get_model()
    except Exception:
        LOGGER.warning("Semantic model load failed, falling back to rule-only grouping")
        return _rule_only_grouping(items)

    cat_queries = _build_category_queries_with_names()
    try:
        q_out = _ensure_cat_queries_encoded(model, cat_queries)
    except Exception as e:
        LOGGER.warning("Failed to encode category queries (%s), fallback to rule-only", e)
        return _rule_only_grouping(items)

    to_score = items[:max_fetch]
    urls = [link for _, link, *_ in to_score]

    texts: Dict[str, str] = {}
    empty_texts = 0
    LOGGER.info("Fetching article texts for semantic scoring: %d candidates", len(urls))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for url, txt in zip(urls, ex.map(fetch_text_cached, urls)):
            if not txt:
                empty_texts += 1
            texts[url] = txt

    kept: Dict[str, List[Tuple[float, str]]] = {name: [] for name, _ in cat_queries}
    score_map: Dict[str, Dict[str, float]] = {name: {} for name, _ in cat_queries}

    for (ts, link, title, summary, domain, why) in to_score:
        text = texts.get(link) or ""
        if not text:
            continue

        try:
            scores = mv_scores_for_all_categories(
                model,
                q_out,
                text,
                mv_weights,
                max_txt,
                chunk_size,
            )
        except Exception as e:
            LOGGER.warning("Semantic scoring error for %s: %s", link, e)
            continue

        for idx, (name, query_str) in enumerate(cat_queries):
            score = float(scores[idx])
            if score >= threshold:
                kept[name].append((score, link))
                score_map.setdefault(name, {})[link] = score
                log_reason(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "link": link,
                        "domain": domain,
                        "category": name,
                        "semantic": True,
                        "score": round(score, 4),
                        "threshold": threshold,
                        "query": query_str,
                        "keyword_hits": (
                            why.get("keywords") if isinstance(why, dict) else None
                        ),
                    }
                )
                break

    grouped = {
        name: [u for s, u in sorted(vals, key=lambda x: x[0], reverse=True)[: CFG["TOPK_PER_CAT"]]]
        for name, vals in kept.items()
    }

    total = sum(len(v) for v in grouped.values())
    if CFG.get("DEBUG_STATS", True):
        LOGGER.info(
            "[debug] semantic: items=%d, to_score=%d, empty_texts=%d, kept_total=%d",
            len(items),
            len(to_score),
            empty_texts,
            total,
        )

    if total == 0 and CFG.get("FALLBACK_IF_EMPTY", True):
        LOGGER.warning("Semantic rerank produced 0 links; falling back to rule-only grouping")
        return _rule_only_grouping(items)

    return grouped, score_map


# ---------------------------------------------------------------------------
# Discovery (Google News)
# ---------------------------------------------------------------------------

def _known_source_domains() -> Set[str]:
    dset: Set[str] = set()
    for s in SOURCES:
        try:
            dset.add(normalize_domain(urlparse(s).netloc))
        except Exception:
            continue
    return dset


def _gnews_search_feeds(query: str) -> List[str]:
    cfg = CFG.get("DISCOVERY", {})
    recency = cfg.get("recency", "when:1d")
    q = quote_plus(f"{query} {recency}")

    feeds: List[str] = []
    if cfg.get("language_only", False):
        lang = cfg.get("lang", "en")
        feeds.append(f"https://news.google.com/rss/search?q={q}&hl={lang}")
    else:
        locales = cfg.get("locales")
        if isinstance(locales, list) and locales:
            for loc in locales:
                hl = loc.get("hl", "en-US")
                gl = loc.get("gl", "US")
                ceid = loc.get("ceid", "US:en")
                feeds.append(
                    f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"
                )
        else:
            hl = cfg.get("hl", "en-US")
            gl = cfg.get("gl", "US")
            ceid = cfg.get("ceid", "US:en")
            feeds.append(
                f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"
            )
    return feeds


def _read_discovered_domains(path: pathlib.Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        return {normalize_domain(x) for x in read_lines(path)}
    except Exception:
        return set()


def _write_discovered_domains(path: pathlib.Path, domains: Set[str]):
    if not domains:
        return
    existing = _read_discovered_domains(path)
    merged = sorted({*existing, *{normalize_domain(d) for d in domains}} - {""})
    try:
        path.write_text(
            "\n".join(merged) + ("\n" if merged else ""),
            encoding="utf-8",
        )
    except Exception as e:  # pragma: no cover
        LOGGER.warning("Discovered domains write error: %s", e)


def discover_outside(
    grouped_existing: Dict[str, List[str]],
    seen_dict: Dict[str, str],
    already_links: Set[str],
):
    """
    Возвращает:
      discovery_links: [url, ...]
      discovery_scores: {url -> score}
    """
    disc_cfg = CFG.get("DISCOVERY", {})
    if not disc_cfg.get("enabled", False):
        return [], {}

    now = int(time.time())
    age_floor = now - AGE_LIMIT_SEC
    known = _known_source_domains()
    cat_queries = [
        (_build_query_for_category(c)) for c in CATEGORIES
    ] or ["mobile security telecom lookup"]

    per_query_limit = int(disc_cfg.get("per_query_limit", 30))
    candidates = []
    for q in cat_queries:
        feed_urls = _gnews_search_feeds(q)
        local_candidates_count = 0
        for feed_url in feed_urls:
            try:
                for ts, has_pub, link, title, summary in parse_feed(feed_url):
                    if CFG.get("REQUIRE_PUBDATE", True) and not has_pub:
                        continue
                    if has_pub and ts is not None and ts < age_floor:
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
                    local_candidates_count += 1
                    if len(candidates) >= per_query_limit * len(cat_queries):
                        break
            except Exception as e:  # pragma: no cover
                LOGGER.warning("Discovery feed error: %s (%s)", feed_url, e)

        if (
            local_candidates_count == 0
            and disc_cfg.get("language_only", False)
            and disc_cfg.get("fallback_ceid_if_empty", False)
        ):
            recency = disc_cfg.get("recency", "when:1d")
            q_enc = quote_plus(f"{q} {recency}")
            for ceid in disc_cfg.get("compat_ceids", ["US:en", "GB:en"]):
                try:
                    country = ceid.split(":")[0]
                    hl = f"en-{country}"
                    feed_url = (
                        f"https://news.google.com/rss/search?q={q_enc}"
                        f"&hl={hl}&gl={country}&ceid={ceid}"
                    )
                    for ts, has_pub, link, title, summary in parse_feed(feed_url):
                        if CFG.get("REQUIRE_PUBDATE", True) and not has_pub:
                            continue
                        if has_pub and ts is not None and ts < age_floor:
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
                except Exception as e:  # pragma: no cover
                    LOGGER.warning("Discovery fallback feed error: %s", e)

    if not candidates:
        return [], {}

    mv_weights = CFG["MV_WEIGHTS"]
    threshold = CFG["THRESHOLD"]
    max_txt = CFG["MAX_TXT_LEN"]
    chunk_size = CFG["CHUNK_SIZE"]
    max_workers = int(CFG.get("MAX_WORKERS", 8))

    # здесь discovery тоже не должен рушить запуск, если модель не грузится
    try:
        model = _get_model()
    except Exception:
        LOGGER.warning("Discovery: semantic model load failed, skipping discovery this run")
        return [], {}

    q_texts = cat_queries
    try:
        q_out = model.encode(
            q_texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
    except Exception as e:
        LOGGER.warning("Discovery: encode of queries failed (%s), skipping", e)
        return [], {}

    urls = [link for _, link, *_ in candidates]
    texts: Dict[str, str] = {}
    LOGGER.info("Discovery: fetching texts for %d candidate articles", len(urls))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for url, txt in zip(urls, ex.map(fetch_text_cached, urls)):
            texts[url] = txt

    scored = []
    for (ts, link, title, summary, dom) in candidates:
        text = texts.get(link) or ""
        if not text:
            continue
        try:
            scores = mv_scores_for_all_categories(
                model,
                q_out,
                text,
                mv_weights,
                max_txt,
                chunk_size,
            )
        except Exception as e:
            LOGGER.debug("Discovery semantic scoring error for %s: %s", link, e)
            continue
        best = max(scores) if scores else 0.0
        best_q = q_texts[scores.index(best)] if scores else None
        if best >= threshold:
            scored.append((best, link, dom, best_q))

    scored.sort(key=lambda x: x[0], reverse=True)
    lim = int(disc_cfg.get("max_links", 10))
    out_links: List[str] = []
    out_domains: Set[str] = set()
    out_scores: Dict[str, float] = {}
    disc_cat_name = disc_cfg.get("category_name", "Outside Sources (Top 10)")

    for s, link, dom, best_q in scored[:lim]:
        out_links.append(link)
        out_domains.add(dom)
        out_scores[link] = float(s)
        log_reason(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "link": link,
                "domain": dom,
                "category": disc_cat_name,
                "semantic": True,
                "score": round(float(s), 4),
                "threshold": threshold,
                "query": best_q,
                "outside": True,
            }
        )

    if disc_cfg.get("save_domains", True):
        _write_discovered_domains(
            ROOT / disc_cfg.get("save_domains_file", "discovered_sources.txt"),
            out_domains,
        )

    return out_links, out_scores


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_outputs(
    grouped: Dict[str, List[str]],
    full_by_cat: Dict[str, List[dict]],
):
    # старый компактный JSON
    atomic_write_text(
        LINKS_JSON,
        json.dumps(grouped, ensure_ascii=False, indent=2),
    )
    # новый «полный» JSON с title, summary, domain, score
    atomic_write_text(
        LINKS_FULL_JSON,
        json.dumps(full_by_cat, ensure_ascii=False, indent=2),
    )
    # текстовый markdown-подобный список
    lines: List[str] = []
    for cat, urls in grouped.items():
        if not urls:
            continue
        lines.append(f"## {cat}")
        lines.extend(urls)
        lines.append("")
    text = ("\n".join(lines).strip() + "\n") if lines else ""
    atomic_write_text(LINKS_TXT, text)


def append_archive(urls: Iterable[str]):
    urls_list = list(urls)
    if not urls_list:
        return
    with open(ARCHIVE_TXT, "a", encoding="utf-8") as f:
        for link in urls_list:
            f.write(link + "\n")


def build_full_records_by_category(
    grouped: Dict[str, List[str]],
    items: List[Tuple[int, str, str, str, str, dict]],
    score_map: Dict[str, Dict[str, float]],
    discovery_scores: Dict[str, float],
) -> Dict[str, List[dict]]:
    """
    Собираем структуру:
      { category -> [ {url,title,summary,domain,score,source}, ... ] }
    """
    items_by_url: Dict[str, Tuple[int, str, str, str, dict]] = {}
    for ts, link, title, summary, domain, why in items:
        items_by_url[link] = (ts, title, summary, domain, why)

    full_by_cat: Dict[str, List[dict]] = {}

    for cat, urls in grouped.items():
        cat_records: List[dict] = []
        for url in urls:
            ts, title, summary, domain, why = items_by_url.get(
                url,
                (None, "", "", "", {}),
            )
            score = score_map.get(cat, {}).get(url)
            if score is None and url in discovery_scores:
                score = discovery_scores[url]

            cat_records.append(
                {
                    "url": url,
                    "title": title,
                    "summary": summary,
                    "domain": domain,
                    "score": score,
                    "source": "discovery" if url in discovery_scores else "rss",
                }
            )
        full_by_cat[cat] = cat_records

    return full_by_cat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    LOGGER.info("Starting news-links aggregation run")
    CACHE_DIR.mkdir(exist_ok=True)

    items = gather()
    seen = load_seen()
    now_iso = datetime.now(timezone.utc).isoformat()

    # фильтруем уже виденные ссылки
    new_items = [it for it in items if it[1] not in seen]

    # основной ранжировщик (semantic + правила, с fallback на правила)
    grouped, score_map = semantic_filter_and_rank(new_items)

    # discovery (Google News) – не роняет скрипт, если что-то пошло не так
    discovery_scores: Dict[str, float] = {}
    try:
        already = {u for urls in grouped.values() for u in urls}
        discovery_links, discovery_scores = discover_outside(
            grouped,
            seen,
            already,
        )
        if discovery_links:
            disc_name = CFG["DISCOVERY"].get(
                "category_name",
                "Outside Sources (Top 10)",
            )
            grouped[disc_name] = discovery_links
    except Exception as e:  # pragma: no cover
        LOGGER.warning("Discovery error: %s", e)

    # собираем полный вывод с метаданными
    full_by_cat = build_full_records_by_category(
        grouped,
        new_items,
        score_map,
        discovery_scores,
    )

    # записываем файлы
    write_outputs(grouped, full_by_cat)

    # обновляем seen и архив
    new_links = {link for _, link, *_ in new_items}
    for urls in grouped.values():
        for link in urls:
            new_links.add(link)

    for link in new_links:
        seen[link] = now_iso
    seen = prune_seen(seen, keep=50000)
    save_seen(seen)
    append_archive(sorted(new_links))

    total_new = sum(len(v) for v in grouped.values())
    LOGGER.info("[OK] New categorized links (incl. discovery): %d", total_new)


if __name__ == "__main__":
    main()
