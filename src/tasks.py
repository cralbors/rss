import hashlib
import inspect
import json
import re
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import feedparser
import numpy as np
import requests
import yaml
from invoke import Context, task as _task
from tqdm import tqdm

FEEDS_PATH = Path("feeds.json")
ENTRIES_DIR = Path("entries")
CONFIG_PATH = Path("config.yaml")
CACHE_DIR = Path("cache")
H_INDEX_CACHE_PATH = CACHE_DIR / "author_h_index.json"

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_CACHE_PATH = CACHE_DIR / "embeddings.npz"
TAG_SIMILARITY_THRESHOLD = 0.35

AWARD_KEYWORD_SET = {"oral", "spotlight", "best paper", "outstanding", "award"}

REQUEST_HEADERS = {
    "User-Agent": "RSS-Aggregator/1.0 (personal feed reader)",
}

NATURE_JOURNAL_DICT = {
    "nature": "nature",
    "nature-genetics": "ng",
    "nature-reviews-genetics": "nrg",
}

CELL_JOURNAL_DICT = {
    "cell": "cell",
    "ajhg": "ajhg",
}

CROSSREF_JOURNAL_DICT = {
    "nucleic-acids-research": "0305-1048",
    "molecular-biology-and-evolution": "0737-4038",
    "bioinformatics": "1367-4803",
}

CROSSREF_API_URL = "https://api.crossref.org"

CROSSREF_CONFERENCE_DICT = {
    "recomb": "RECOMB",
    "ismb": "ISMB",
}

OPENREVIEW_VENUE_DICT = {
    "neurips": "NeurIPS.cc",
    "icml": "ICML.cc",
    "iclr": "ICLR.cc",
}

OPENREVIEW_API_URL = "https://api2.openreview.net"


def task(func):
    """ Task decorator that handles context parameter automatically """

    sig = inspect.signature(func)

    param_list = [inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Context)]
    param_list.extend(sig.parameters.values())
    new_sig = sig.replace(parameters=param_list)

    def wrapped_func(ctx, *args, **kwargs):
        return func(*args, **kwargs)

    wrapped_func.__name__ = func.__name__
    wrapped_func.__doc__ = func.__doc__
    wrapped_func.__module__ = func.__module__
    wrapped_func.__signature__ = new_sig

    task_object = _task(wrapped_func)
    task_object.func = func

    return task_object


def _load_feed_list() -> list[dict]:
    """ Load feeds from `feeds.json` """

    if FEEDS_PATH.exists():
        with open(FEEDS_PATH, "r") as f:
            return json.load(f)
    return []


def _save_feed_list(feed_dict_list: list[dict]) -> None:
    """ Save feeds to `feeds.json` """

    with open(FEEDS_PATH, "w") as f:
        json.dump(feed_dict_list, f, indent=2)


def _load_entry_list(feed_name: str) -> list[dict]:
    """ Load stored entries for a feed """

    path = ENTRIES_DIR / f"{feed_name}.json"

    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []


def _save_entry_list(feed_name: str, entry_dict_list: list[dict]) -> None:
    """ Save entries for a feed """

    ENTRIES_DIR.mkdir(exist_ok=True)
    path = ENTRIES_DIR / f"{feed_name}.json"

    with open(path, "w") as f:
        json.dump(entry_dict_list, f, indent=2)


def _strip_html(text: str) -> str:
    """ Remove HTML tags from text """

    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _extract_doi(url: str) -> str | None:
    """ Extract DOI from a URL if present """

    doi_match = re.search(r"(?:doi\.org/|/doi/)(10\.\d{4,}/[^\s]+)", url)

    if doi_match:
        return doi_match.group(1).rstrip(".")

    return None


def _normalize_title(title: str) -> str:
    """ Normalize a title for fuzzy matching """

    normalized = title.lower()
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _entry_richness(entry: dict) -> int:
    """ Score how much metadata an entry has (higher = richer) """

    score = 0
    score += len(entry.get("summary", ""))
    score += 100 if entry.get("last_author") else 0
    score += 100 if entry.get("awarded") else 0
    score += 50 if entry.get("venue") else 0
    score += 50 if entry.get("published") else 0
    return score


def _deduplicate_entry_list(entry_dict_list: list[dict]) -> list[dict]:
    """ Deduplicate entries across feeds by DOI or normalized title """

    seen_doi_dict: dict[str, int] = {}
    seen_title_dict: dict[str, int] = {}
    result_entry_list: list[dict] = []

    for entry in entry_dict_list:
        doi = _extract_doi(entry.get("url", ""))
        norm_title = _normalize_title(entry.get("title", ""))

        if not norm_title:
            result_entry_list.append(entry)
            continue

        duplicate_idx = None

        if doi and doi in seen_doi_dict:
            duplicate_idx = seen_doi_dict[doi]
        elif norm_title and norm_title in seen_title_dict:
            duplicate_idx = seen_title_dict[norm_title]

        if duplicate_idx is not None:
            existing = result_entry_list[duplicate_idx]

            if _entry_richness(entry) > _entry_richness(existing):
                result_entry_list[duplicate_idx] = entry
        else:
            idx = len(result_entry_list)
            result_entry_list.append(entry)

            if doi:
                seen_doi_dict[doi] = idx

            if norm_title:
                seen_title_dict[norm_title] = idx

    return result_entry_list


def _parse_rss_entry(entry, feed_name: str) -> dict:
    """ Parse a feedparser entry into a normalized dict """

    published = ""

    if hasattr(entry, "published_parsed") and entry.published_parsed:
        published = datetime(*entry.published_parsed[:6]).isoformat()
    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
        published = datetime(*entry.updated_parsed[:6]).isoformat()

    summary = _strip_html(getattr(entry, "summary", ""))

    if len(summary) > 500:
        summary = summary[:500] + "..."

    return {
        "title": _strip_html(getattr(entry, "title", "")),
        "url": getattr(entry, "link", ""),
        "published": published,
        "summary": summary,
        "feed_name": feed_name,
        "fetched_date": datetime.now().isoformat(),
    }


def _deduplicate_and_store(feed_name: str, new_entry_list: list[dict]) -> list[dict]:
    """ Deduplicate new entries against existing ones and save """

    existing_entry_list = _load_entry_list(feed_name)
    existing_url_set = {entry["url"] for entry in existing_entry_list}

    unique_entry_list = []

    for entry_dict in new_entry_list:
        if entry_dict["url"] and entry_dict["url"] not in existing_url_set:
            unique_entry_list.append(entry_dict)
            existing_url_set.add(entry_dict["url"])

    if unique_entry_list:
        all_entry_list = existing_entry_list + unique_entry_list
        _save_entry_list(feed_name, all_entry_list)

    return unique_entry_list


def _fetch_edgar_feed(feed_dict: dict) -> list[dict]:
    """ Fetch recent SEC filings via EDGAR submissions API """

    cik = feed_dict["url"]

    try:
        response = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=REQUEST_HEADERS,
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  warning: edgar request failed for `{feed_dict['name']}`: {e}")
        return []

    data = response.json()
    recent = data.get("filings", {}).get("recent", {})
    form_list = recent.get("form", [])
    date_list = recent.get("filingDate", [])
    accession_list = recent.get("accessionNumber", [])
    document_list = recent.get("primaryDocument", [])
    description_list = recent.get("primaryDocDescription", [])

    cik_number = cik.lstrip("0")
    new_entry_list = []

    for i in range(len(form_list)):
        if form_list[i] not in ("10-K", "10-Q"):
            continue

        accession_clean = accession_list[i].replace("-", "")
        doc = document_list[i]
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_number}/{accession_clean}/{doc}"

        description = description_list[i] if i < len(description_list) else ""

        entry_dict = {
            "title": f"{data.get('name', feed_dict['name'])} - {form_list[i]} ({date_list[i]})",
            "url": url,
            "published": date_list[i],
            "summary": description,
            "feed_name": feed_dict["name"],
            "fetched_date": datetime.now().isoformat(),
        }
        new_entry_list.append(entry_dict)

    return _deduplicate_and_store(feed_dict["name"], new_entry_list)


def _fetch_rss_feed(feed_dict: dict) -> list[dict]:
    """ Fetch and parse entries from an RSS/Atom feed """

    try:
        response = requests.get(feed_dict["url"], headers=REQUEST_HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  warning: request failed for `{feed_dict['name']}`: {e}")
        return []

    parsed = feedparser.parse(response.content)

    if parsed.bozo and not parsed.entries:
        print(f"  warning: failed to parse feed `{feed_dict['name']}`")
        return []

    new_entry_list = [_parse_rss_entry(entry, feed_dict["name"]) for entry in parsed.entries]
    return _deduplicate_and_store(feed_dict["name"], new_entry_list)


def _fetch_openreview_feed(feed_dict: dict) -> list[dict]:
    """ Fetch accepted papers from OpenReview for a conference venue """

    venue_prefix = feed_dict["url"]
    now = datetime.now()
    year = now.year

    if now.month < 6:
        year -= 1

    new_entry_list = []

    for search_year in [year, year - 1]:
        venue_id = f"{venue_prefix}/{search_year}/Conference"

        try:
            response = requests.get(
                f"{OPENREVIEW_API_URL}/notes",
                params={"content.venueid": venue_id, "limit": 200},
                headers=REQUEST_HEADERS,
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  warning: openreview request failed for `{feed_dict['name']}` {search_year}: {e}")
            continue

        data = response.json()

        for note in data.get("notes", []):
            content = note.get("content", {})
            title_field = content.get("title", {})
            title = title_field.get("value", "") if isinstance(title_field, dict) else str(title_field)

            abstract_field = content.get("abstract", {})
            abstract = abstract_field.get("value", "") if isinstance(abstract_field, dict) else str(abstract_field)

            if len(abstract) > 500:
                abstract = abstract[:500] + "..."

            venue_field = content.get("venue", {})
            venue = venue_field.get("value", "") if isinstance(venue_field, dict) else str(venue_field)

            paper_url = f"https://openreview.net/forum?id={note['id']}"
            published = note.get("cdate", "")

            if published:
                published = datetime.fromtimestamp(published / 1000).isoformat()

            authors_field = content.get("authors", {})
            author_list = authors_field.get("value", []) if isinstance(authors_field, dict) else []
            last_author = author_list[-1] if author_list else ""

            venue_lower = venue.lower()
            paper_is_awarded = any(kw in venue_lower for kw in AWARD_KEYWORD_SET)
            paper_is_poster = "poster" in venue_lower

            entry_dict = {
                "title": title,
                "url": paper_url,
                "published": published,
                "summary": abstract,
                "venue": venue,
                "last_author": last_author,
                "awarded": paper_is_awarded,
                "poster": paper_is_poster,
                "feed_name": feed_dict["name"],
                "fetched_date": datetime.now().isoformat(),
            }
            new_entry_list.append(entry_dict)

    return _deduplicate_and_store(feed_dict["name"], new_entry_list)


def _parse_crossref_item(item: dict, feed_name: str) -> dict:
    """ Parse a Crossref work item into a normalized entry dict """

    title_list = item.get("title", [])
    title = _strip_html(title_list[0]) if title_list else ""

    abstract = _strip_html(item.get("abstract", ""))

    if len(abstract) > 500:
        abstract = abstract[:500] + "..."

    doi = item.get("DOI", "")
    url = f"https://doi.org/{doi}" if doi else ""

    published = ""
    date_parts = item.get("published", {}).get("date-parts", [[]])

    if date_parts and date_parts[0]:
        parts = date_parts[0]
        year = str(parts[0]) if len(parts) > 0 else ""
        month = str(parts[1]).zfill(2) if len(parts) > 1 else "01"
        day = str(parts[2]).zfill(2) if len(parts) > 2 else "01"
        published = f"{year}-{month}-{day}"

    return {
        "title": title,
        "url": url,
        "published": published,
        "summary": abstract,
        "feed_name": feed_name,
        "fetched_date": datetime.now().isoformat(),
    }


def _fetch_crossref_feed(feed_dict: dict) -> list[dict]:
    """ Fetch recent papers from a journal via Crossref ISSN lookup """

    issn = feed_dict["url"]

    try:
        response = requests.get(
            f"{CROSSREF_API_URL}/journals/{issn}/works",
            params={"rows": 25, "sort": "published", "order": "desc"},
            headers={"User-Agent": "RSS-Aggregator/1.0 (mailto:research@example.com)"},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  warning: crossref request failed for `{feed_dict['name']}`: {e}")
        return []

    data = response.json()
    new_entry_list = [
        _parse_crossref_item(item, feed_dict["name"])
        for item in data.get("message", {}).get("items", [])
    ]

    return _deduplicate_and_store(feed_dict["name"], new_entry_list)


def _fetch_crossref_query_feed(feed_dict: dict) -> list[dict]:
    """ Fetch conference proceedings via Crossref search query """

    query = feed_dict["url"]

    try:
        response = requests.get(
            f"{CROSSREF_API_URL}/works",
            params={
                "query": query,
                "filter": "type:book-chapter",
                "rows": 25,
                "sort": "published",
                "order": "desc",
            },
            headers={"User-Agent": "RSS-Aggregator/1.0 (mailto:research@example.com)"},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  warning: crossref query failed for `{feed_dict['name']}`: {e}")
        return []

    data = response.json()
    new_entry_list = [
        _parse_crossref_item(item, feed_dict["name"])
        for item in data.get("message", {}).get("items", [])
    ]

    return _deduplicate_and_store(feed_dict["name"], new_entry_list)


def _fetch_single_feed(feed_dict: dict) -> list[dict]:
    """ Dispatch to the right fetcher based on feed type """

    feed_type = feed_dict.get("feed_type", "rss")

    if feed_type == "edgar":
        return _fetch_edgar_feed(feed_dict)
    elif feed_type == "openreview":
        return _fetch_openreview_feed(feed_dict)
    elif feed_type == "crossref":
        return _fetch_crossref_feed(feed_dict)
    elif feed_type == "crossref_query":
        return _fetch_crossref_query_feed(feed_dict)
    else:
        return _fetch_rss_feed(feed_dict)


@task
def add_feed(url: str, name: str, category: str = "", feed_type: str = "rss"):
    """ Add a new RSS feed to `feeds.json` """

    feed_dict_list = _load_feed_list()

    existing_url_set = {feed_dict["url"] for feed_dict in feed_dict_list}
    existing_name_set = {feed_dict["name"].lower() for feed_dict in feed_dict_list}

    if url in existing_url_set:
        print(f"URL already exists for feed `{name}`")
        return

    if name.lower() in existing_name_set:
        print(f"Feed `{name}` already exists")
        return

    feed_dict = {
        "url": url,
        "name": name,
        "category": category,
        "feed_type": feed_type,
        "added_date": datetime.now().isoformat(),
    }

    feed_dict_list.append(feed_dict)
    _save_feed_list(feed_dict_list)
    print(f"Added feed `{name}`" + (f" [{category}]" if category else ""))


@task
def remove_feed(name: str):
    """ Remove an RSS feed by name """

    feed_dict_list = _load_feed_list()

    if not feed_dict_list:
        return

    idx = None

    for i, feed_dict in enumerate(feed_dict_list):
        if feed_dict["name"].lower() == name.lower():
            idx = i
            break

    if idx is None:
        print(f"Feed `{name}` not found")
        return

    feed_dict_list.pop(idx)
    _save_feed_list(feed_dict_list)
    print(f"Removed feed `{name}`")


@task
def list_feeds():
    """ List all registered feeds """

    feed_dict_list = _load_feed_list()

    if not feed_dict_list:
        print("No feeds registered")
        return

    max_name_len = max(len(f["name"]) for f in feed_dict_list)
    max_cat_len = max(len(f.get("category", "")) for f in feed_dict_list)
    max_type_len = max(len(f.get("feed_type", "rss")) for f in feed_dict_list)

    for feed_dict in feed_dict_list:
        name = feed_dict["name"].ljust(max_name_len)
        category = feed_dict.get("category", "").ljust(max_cat_len)
        feed_type = feed_dict.get("feed_type", "rss").ljust(max_type_len)
        url = feed_dict["url"][:70]
        print(f"  {name}  {category}  {feed_type}  {url}")

    print(f"\n{len(feed_dict_list)} feeds total\n")


@task
def add_edgar_feed(name: str, cik: str):
    """ Add SEC EDGAR feed for a company by CIK (fetches 10-K and 10-Q filings) """

    cik = cik.zfill(10)
    add_feed.func(cik, name, category="filings", feed_type="edgar")


@task
def add_nature_feed(journal: str):
    """
    Add a Nature/Springer journal RSS feed

    Available journals: nature, nature-genetics, nature-reviews-genetics
    """

    journal_lower = journal.lower()

    if journal_lower not in NATURE_JOURNAL_DICT:
        available = ", ".join(NATURE_JOURNAL_DICT.keys())
        print(f"Unknown journal `{journal}`. Available: {available}")
        return

    code = NATURE_JOURNAL_DICT[journal_lower]
    url = f"https://www.nature.com/{code}.rss"
    add_feed.func(url, journal_lower, category="journals")


@task
def add_cell_feed(journal: str = "cell"):
    """
    Add a Cell Press journal RSS feed

    Available journals: cell, ajhg
    """

    journal_lower = journal.lower()

    if journal_lower not in CELL_JOURNAL_DICT:
        available = ", ".join(CELL_JOURNAL_DICT.keys())
        print(f"Unknown journal `{journal}`. Available: {available}")
        return

    slug = CELL_JOURNAL_DICT[journal_lower]
    url = f"https://www.cell.com/{slug}/current.rss"
    add_feed.func(url, journal_lower, category="journals")


@task
def add_science_feed():
    """ Add Science magazine RSS feed """

    url = "https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=science"
    add_feed.func(url, "science", category="journals")


@task
def add_crossref_feed(journal: str):
    """
    Add a journal feed via Crossref (official DOI registry)

    Available: nucleic-acids-research, molecular-biology-and-evolution, bioinformatics
    """

    journal_lower = journal.lower()

    if journal_lower not in CROSSREF_JOURNAL_DICT:
        available = ", ".join(CROSSREF_JOURNAL_DICT.keys())
        print(f"Unknown journal `{journal}`. Available: {available}")
        return

    issn = CROSSREF_JOURNAL_DICT[journal_lower]
    add_feed.func(issn, journal_lower, category="journals", feed_type="crossref")


@task
def add_conference_feed(conference: str):
    """
    Add a conference feed via OpenReview (ML) or Crossref (bio)

    ML conferences (via OpenReview): neurips, icml, iclr
    Bio conferences (via Crossref): recomb, ismb
    """

    conference_lower = conference.lower()

    if conference_lower in OPENREVIEW_VENUE_DICT:
        venue_prefix = OPENREVIEW_VENUE_DICT[conference_lower]
        add_feed.func(venue_prefix, conference_lower, category="conferences", feed_type="openreview")
    elif conference_lower in CROSSREF_CONFERENCE_DICT:
        query = CROSSREF_CONFERENCE_DICT[conference_lower]
        add_feed.func(query, conference_lower, category="conferences", feed_type="crossref_query")
    else:
        all_available = ", ".join(list(OPENREVIEW_VENUE_DICT.keys()) + list(CROSSREF_CONFERENCE_DICT.keys()))
        print(f"Unknown conference `{conference}`. Available: {all_available}")


@task
def fetch_feeds(name: str = ""):
    """ Fetch new entries from all feeds (or a specific feed by name) """

    feed_dict_list = _load_feed_list()

    if not feed_dict_list:
        print("No feeds registered")
        return

    if name:
        feed_dict_list = [f for f in feed_dict_list if f["name"].lower() == name.lower()]

        if not feed_dict_list:
            print(f"Feed `{name}` not found")
            return

    total_new = 0

    for feed_dict in tqdm(feed_dict_list, desc="Fetching feeds"):
        new_entry_list = _fetch_single_feed(feed_dict)
        count = len(new_entry_list)

        if count > 0:
            tqdm.write(f"  {feed_dict['name']}: {count} new entries")

        total_new += count

    print(f"\n{total_new} new entries across {len(feed_dict_list)} feeds\n")


@task
def show_new(hours: int = 24):
    """ Show entries fetched within the last N hours (default 24) """

    cutoff = datetime.now() - timedelta(hours=hours)
    feed_dict_list = _load_feed_list()

    if not feed_dict_list:
        print("No feeds registered")
        return

    new_entry_list = []

    for feed_dict in feed_dict_list:
        entry_list = _load_entry_list(feed_dict["name"])

        for entry in entry_list:
            fetched = entry.get("fetched_date", "")

            if fetched and datetime.fromisoformat(fetched) > cutoff:
                new_entry_list.append(entry)

    if not new_entry_list:
        print(f"No new entries in the last {hours} hours")
        return

    deduped_count = len(new_entry_list)
    new_entry_list = _deduplicate_entry_list(new_entry_list)
    deduped_count -= len(new_entry_list)

    new_entry_list.sort(key=lambda e: e.get("fetched_date", ""), reverse=True)

    current_feed = ""

    for entry in new_entry_list:
        if entry["feed_name"] != current_feed:
            current_feed = entry["feed_name"]
            print(f"\n--- {current_feed} ---")

        title = entry["title"][:100]
        published = entry.get("published", "")[:10]
        url = entry["url"]
        print(f"  [{published}] {title}")
        print(f"           {url}")

    dedup_msg = f" ({deduped_count} duplicates removed)" if deduped_count > 0 else ""
    print(f"\n{len(new_entry_list)} new entries in the last {hours} hours{dedup_msg}\n")


def _load_interest_list() -> list[str]:
    """ Load interest descriptions from `config.yaml` """

    if not CONFIG_PATH.exists():
        return []

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}

    return config.get("interests", [])


def _load_h_index_cache() -> dict[str, dict]:
    """ Load cached author h-index lookups """

    if H_INDEX_CACHE_PATH.exists():
        with open(H_INDEX_CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_h_index_cache(cache_dict: dict[str, dict]) -> None:
    """ Save author h-index cache """

    CACHE_DIR.mkdir(exist_ok=True)

    with open(H_INDEX_CACHE_PATH, "w") as f:
        json.dump(cache_dict, f, indent=2)


def _lookup_author_h_index(author_name: str, cache_dict: dict[str, dict]) -> int | None:
    """ Look up an author's h-index via Semantic Scholar, with caching """

    if author_name in cache_dict:
        return cache_dict[author_name].get("h_index")

    try:
        response = requests.get(
            f"{SEMANTIC_SCHOLAR_API_URL}/author/search",
            params={"query": author_name, "limit": 1, "fields": "name,hIndex"},
            timeout=10,
        )

        if response.status_code == 429:
            return None

        response.raise_for_status()
    except requests.RequestException:
        return None

    data = response.json()
    author_list = data.get("data", [])

    if not author_list:
        cache_dict[author_name] = {"h_index": None, "fetched_date": datetime.now().isoformat()}
        return None

    h_index = author_list[0].get("h_index")
    cache_dict[author_name] = {"h_index": h_index, "fetched_date": datetime.now().isoformat()}
    return h_index


def _enrich_h_index(entry_dict_list: list[dict]) -> None:
    """ Batch-enrich entries with last author h-index (modifies in place) """

    author_name_set = set()

    for entry in entry_dict_list:
        author = entry.get("last_author", "")

        if author and "h_index" not in entry:
            author_name_set.add(author)

    if not author_name_set:
        return

    cache_dict = _load_h_index_cache()
    uncached_name_list = [name for name in author_name_set if name not in cache_dict]

    if uncached_name_list:
        for name in tqdm(uncached_name_list, desc="Looking up h-indices", leave=False):
            _lookup_author_h_index(name, cache_dict)

        _save_h_index_cache(cache_dict)

    for entry in entry_dict_list:
        author = entry.get("last_author", "")

        if author and author in cache_dict:
            entry["h_index"] = cache_dict[author].get("h_index")


_embedding_model = None


def _get_embedding_model():
    """ Lazy-load the sentence-transformers model """

    global _embedding_model

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _embedding_model


def _hash_text(text: str) -> str:
    """ Hash a string for cache keying """

    return hashlib.md5(text.encode()).hexdigest()


def _load_embedding_cache() -> dict[str, np.ndarray]:
    """ Load cached embeddings from disk """

    if EMBEDDING_CACHE_PATH.exists():
        data = np.load(EMBEDDING_CACHE_PATH, allow_pickle=True)
        return dict(data)
    return {}


def _save_embedding_cache(cache_dict: dict[str, np.ndarray]) -> None:
    """ Save embedding cache to disk """

    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(EMBEDDING_CACHE_PATH, **cache_dict)


def _embed_text_list_cached(text_list: list[str], cache_dict: dict[str, np.ndarray]) -> np.ndarray:
    """ Encode strings with caching — only encodes uncached texts """

    hash_list = [_hash_text(t) for t in text_list]

    uncached_idx_list = [i for i, h in enumerate(hash_list) if h not in cache_dict]

    if uncached_idx_list:
        uncached_text_list = [text_list[i] for i in uncached_idx_list]
        model = _get_embedding_model()
        new_embedding_array = model.encode(
            uncached_text_list, show_progress_bar=len(uncached_text_list) > 50, normalize_embeddings=True,
        )

        for j, idx in enumerate(uncached_idx_list):
            cache_dict[hash_list[idx]] = new_embedding_array[j]

    embedding_list = [cache_dict[h] for h in hash_list]
    return np.stack(embedding_list)


def _get_entry_text(entry: dict) -> str:
    """ Build text representation of an entry for embedding """

    return f"{entry.get('title', '')} {entry.get('summary', '')}"


def _get_relevance_score_list(
    entry_dict_list: list[dict],
    interest_list: list[str],
    embedding_cache_dict: dict[str, np.ndarray],
) -> list[float]:
    """ Semantic similarity against interest descriptions via sentence-transformers """

    if not entry_dict_list or not interest_list:
        return [0.0] * len(entry_dict_list)

    entry_text_list = [_get_entry_text(entry) for entry in entry_dict_list]

    interest_embedding_array = _embed_text_list_cached(interest_list, embedding_cache_dict)
    entry_embedding_array = _embed_text_list_cached(entry_text_list, embedding_cache_dict)

    similarity_matrix = entry_embedding_array @ interest_embedding_array.T
    score_array = similarity_matrix.max(axis=1)

    return score_array.tolist()


def _load_tag_list() -> list[str]:
    """ Load tag definitions from `config.yaml` """

    if not CONFIG_PATH.exists():
        return []

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}

    return config.get("tags", [])


def _get_tag_list_for_entry(
    entry_embedding: np.ndarray,
    tag_list: list[str],
    tag_embedding_array: np.ndarray,
    threshold: float = TAG_SIMILARITY_THRESHOLD,
) -> list[str]:
    """ Assign tags to an entry based on cosine similarity threshold """

    similarity_array = entry_embedding @ tag_embedding_array.T
    matched_tag_list = []

    for i, score in enumerate(similarity_array):
        if score >= threshold:
            matched_tag_list.append(tag_list[i])

    return matched_tag_list


def _enrich_tags(entry_dict_list: list[dict], embedding_cache_dict: dict[str, np.ndarray]) -> None:
    """ Batch-assign tags to entries based on semantic similarity (modifies in place) """

    tag_list = _load_tag_list()

    if not tag_list:
        return

    entry_text_list = [_get_entry_text(entry) for entry in entry_dict_list]

    tag_embedding_array = _embed_text_list_cached(tag_list, embedding_cache_dict)
    entry_embedding_array = _embed_text_list_cached(entry_text_list, embedding_cache_dict)

    for i, entry in enumerate(entry_dict_list):
        entry["tags"] = _get_tag_list_for_entry(
            entry_embedding_array[i], tag_list, tag_embedding_array,
        )


def _get_composite_score_list(
    entry_dict_list: list[dict],
    relevance_score_list: list[float],
    h_index_weight: float = 0.3,
    award_boost: float = 0.2,
) -> list[float]:
    """
    Combine relevance, h-index, and award status into a single score

    score = relevance * (1 - h_index_weight) + normalized_h_index * h_index_weight + award_boost
    """

    h_index_list = []

    for entry in entry_dict_list:
        h = entry.get("h_index")
        h_index_list.append(h if h is not None else 0)

    max_h_index = max(h_index_list) if h_index_list else 1
    max_h_index = max(max_h_index, 1)

    composite_score_list = []

    for i, entry in enumerate(entry_dict_list):
        relevance = relevance_score_list[i]
        h_index_normalized = h_index_list[i] / max_h_index

        has_h_index = entry.get("last_author", "") != ""

        if has_h_index:
            score = relevance * (1 - h_index_weight) + h_index_normalized * h_index_weight
        else:
            score = relevance

        if entry.get("awarded", False):
            score += award_boost

        composite_score_list.append(score)

    return composite_score_list


def _get_award_tag(entry: dict) -> str:
    """ Get display tag for awarded papers """

    if not entry.get("awarded", False):
        return ""

    venue_lower = entry.get("venue", "").lower()

    if "oral" in venue_lower:
        return "ORAL"
    elif "spotlight" in venue_lower:
        return "SPOTLIGHT"

    return "AWARDED"


def _build_scored_entry_list(
    hours: int,
    interest_list: list[str],
    is_awarded: bool = False,
) -> list[tuple[float, dict]] | None:
    """ Collect, dedup, enrich, and score entries """

    cutoff = datetime.now() - timedelta(hours=hours)
    feed_dict_list = _load_feed_list()

    if not feed_dict_list:
        print("No feeds registered")
        return None

    candidate_entry_list = []

    for feed_dict in feed_dict_list:
        entry_list = _load_entry_list(feed_dict["name"])

        for entry in entry_list:
            fetched = entry.get("fetched_date", "")

            if fetched and datetime.fromisoformat(fetched) > cutoff:
                candidate_entry_list.append(entry)

    if not candidate_entry_list:
        print(f"No entries in the last {hours} hours")
        return None

    candidate_entry_list = _deduplicate_entry_list(candidate_entry_list)

    if is_awarded:
        candidate_entry_list = [e for e in candidate_entry_list if e.get("awarded", False)]

        if not candidate_entry_list:
            print("No awarded entries found")
            return None

    _enrich_h_index(candidate_entry_list)

    embedding_cache_dict = _load_embedding_cache()
    _enrich_tags(candidate_entry_list, embedding_cache_dict)
    relevance_score_list = _get_relevance_score_list(candidate_entry_list, interest_list, embedding_cache_dict)
    _save_embedding_cache(embedding_cache_dict)

    composite_score_list = _get_composite_score_list(candidate_entry_list, relevance_score_list)

    scored_entry_list = list(zip(composite_score_list, candidate_entry_list))
    scored_entry_list.sort(key=lambda x: x[0], reverse=True)

    return scored_entry_list


def _render_digest_html(scored_entry_list: list[tuple[float, dict]], top_n: int, hours: int, is_awarded: bool) -> str:
    """ Render scored entries as an HTML page """

    display_count = min(top_n, len(scored_entry_list))
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    label = "Awarded Papers" if is_awarded else "Feed Digest"

    rows = []

    for rank, (score, entry) in enumerate(scored_entry_list[:display_count], 1):
        title = entry["title"]
        url = entry.get("url", "")
        published = entry.get("published", "")[:10]
        feed = entry["feed_name"]
        summary = entry.get("summary", "")[:200]
        last_author = entry.get("last_author", "")
        h_index = entry.get("h_index")
        award_tag = _get_award_tag(entry)

        meta_part_list = [feed, published]

        if last_author:
            author_str = last_author

            if h_index is not None:
                author_str += f" (h={h_index})"

            meta_part_list.append(author_str)

        meta = " | ".join(p for p in meta_part_list if p)
        award_html = f'<span class="award">{award_tag}</span> ' if award_tag else ""

        tag_list = entry.get("tags", [])
        tag_html = " ".join(f'<span class="tag">{t}</span>' for t in tag_list)

        rows.append(f"""<tr>
<td class="rank">{rank}</td>
<td class="score">{score:.2f}</td>
<td>{award_html}<a href="{url}">{title}</a>
<div class="meta">{meta}</div>
<div class="tags">{tag_html}</div>
<div class="summary">{summary}</div></td>
</tr>""")

    table_rows = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{label} - {now}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
h1 {{ font-size: 1.4em; border-bottom: 2px solid #333; padding-bottom: 8px; }}
.subtitle {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
table {{ width: 100%; border-collapse: collapse; }}
tr {{ border-bottom: 1px solid #eee; }}
tr:hover {{ background: #f8f8f8; }}
td {{ padding: 10px 8px; vertical-align: top; }}
.rank {{ width: 30px; color: #999; font-weight: bold; text-align: right; }}
.score {{ width: 40px; color: #666; font-family: monospace; }}
a {{ color: #1a0dab; text-decoration: none; font-weight: 500; }}
a:hover {{ text-decoration: underline; }}
.meta {{ color: #666; font-size: 0.85em; margin-top: 3px; }}
.summary {{ color: #888; font-size: 0.8em; margin-top: 3px; }}
.award {{ background: #ff6b00; color: white; padding: 1px 6px; border-radius: 3px; font-size: 0.75em; font-weight: bold; }}
.tags {{ margin-top: 3px; }}
.tag {{ background: #e8f0fe; color: #1967d2; padding: 1px 6px; border-radius: 3px; font-size: 0.72em; margin-right: 4px; }}
</style>
</head><body>
<h1>{label}</h1>
<div class="subtitle">Generated {now} | {display_count}/{len(scored_entry_list)} entries | Last {hours}h</div>
<table>{table_rows}</table>
</body></html>"""


DIGEST_DIR = Path("digests")


def _load_email() -> str | None:
    """ Load email address from `config.yaml` """

    if not CONFIG_PATH.exists():
        return None

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}

    return config.get("email")


def _send_digest_email(html_content: str, recipient: str, subject: str) -> None:
    """ Send HTML digest via local sendmail """

    msg = MIMEMultipart("alternative")
    msg["From"] = f"RSS Digest <{recipient}>"
    msg["To"] = recipient
    msg["Subject"] = subject

    plain_text = "Your RSS digest is attached as HTML. View in an HTML-capable email client."
    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP("localhost") as server:
        server.sendmail(recipient, [recipient], msg.as_string())


@task
def digest(hours: int = 168, top_n: int = 30, is_awarded: bool = False, save_html: bool = False, send_email: bool = False):
    """
    Show top entries ranked by relevance, h-index, and awards

    --is-awarded: only oral/spotlight/best paper. --save-html: save to digests/. --send-email: send to configured address.
    """

    interest_list = _load_interest_list()

    if not interest_list:
        print("No interests defined in config.yaml")
        return

    scored_entry_list = _build_scored_entry_list(hours, interest_list, is_awarded=is_awarded)

    if scored_entry_list is None:
        return

    display_count = min(top_n, len(scored_entry_list))

    if save_html or send_email:
        html_content = _render_digest_html(scored_entry_list, top_n, hours, is_awarded)

        if save_html:
            DIGEST_DIR.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            file_name = f"digest_{timestamp}.html"
            output_path = DIGEST_DIR / file_name

            with open(output_path, "w") as f:
                f.write(html_content)

            print(f"Saved digest to {output_path}")

        if send_email:
            recipient = _load_email()

            if not recipient:
                print("No email configured in config.yaml")
                return

            now = datetime.now().strftime("%Y-%m-%d")
            label = "Awarded Papers" if is_awarded else "Feed Digest"
            subject = f"{label} - {now}"

            try:
                _send_digest_email(html_content, recipient, subject)
                print(f"Sent digest to {recipient}")
            except Exception as e:
                print(f"Failed to send email: {e}")

        return

    for rank, (score, entry) in enumerate(scored_entry_list[:display_count], 1):
        title = entry["title"][:85]
        published = entry.get("published", "")[:10]
        feed = entry["feed_name"]
        url = entry["url"]
        h_index = entry.get("h_index")
        last_author = entry.get("last_author", "")
        award_tag = _get_award_tag(entry)

        award_str = f" {award_tag}" if award_tag else ""
        h_index_tag = f" h={h_index}" if h_index is not None else ""
        author_tag = f" {last_author}" if last_author else ""
        tag_list = entry.get("tags", [])
        tag_str = f" [{', '.join(tag_list)}]" if tag_list else ""

        print(f"  {rank:>3}. [{score:.2f}]{award_str} {title}")
        print(f"       {feed} | {published} |{author_tag}{h_index_tag}{tag_str} | {url}")

    total = len(scored_entry_list)
    label = "awarded " if is_awarded else ""
    print(f"\n{display_count}/{total} {label}entries shown (last {hours}h)\n")
