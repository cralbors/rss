# rss

Personal intelligence aggregator for computational biology, AI, and biotech. Fetches papers, filings, and conference proceedings from multiple sources, scores them by relevance using sentence-transformer embeddings, and generates prioritized digests.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

All commands use [invoke](https://www.pyinvoke.org/):

```bash
# List available tasks
invoke --list

# Add feeds
invoke add-nature-feed nature-genetics
invoke add-cell-feed cell
invoke add-science-feed
invoke add-crossref-feed nucleic-acids-research
invoke add-conference-feed neurips
invoke add-edgar-feed regeneron --cik 0000872589

# Fetch all feeds
invoke fetch-feeds

# Show recent entries
invoke show-new --hours 48

# Generate scored digest (terminal)
invoke digest --top-n 20

# Generate HTML digest
invoke digest --save-html --top-n 50

# Filter to awarded papers only
invoke digest --is-awarded
```

## Feed backends

| Type | Source | Examples |
|------|--------|----------|
| `rss` | RSS/Atom | Nature, Cell, Science, AJHG |
| `crossref` | Crossref API (ISSN) | NAR, Bioinformatics, MBE |
| `crossref_query` | Crossref API (search) | RECOMB, ISMB |
| `openreview` | OpenReview API | NeurIPS, ICML, ICLR |
| `edgar` | SEC EDGAR API | Regeneron, Lilly, Vertex |

## Scoring

Entries are ranked by a composite score:

- **Relevance** (70%): cosine similarity between entry text and interest profile, computed with `all-MiniLM-L6-v2` sentence-transformer embeddings
- **Author h-index** (30%): last author's h-index via Semantic Scholar, normalized across the entry set
- **Award boost** (+0.2): for oral, spotlight, or best paper designations

## Tagging

Entries are auto-tagged by semantic similarity against predefined tags in `config.yaml`. Tags appear in both terminal and HTML digest output.

## Configuration

Edit `config.yaml` to customize:

```yaml
email: "user@example.com"

interests:
  - "CRISPR gene editing therapy drug target identification"
  - "protein structure prediction AlphaFold"
  - "genomics variant effect prediction"

tags:
  - "gwas"
  - "single-cell"
  - "nlp"
  - "genomics"
```

## Caching

- `cache/author_h_index.json`: Semantic Scholar h-index lookups
- `cache/embeddings.npz`: sentence-transformer embeddings for entries, tags, and interests

Embedding cache makes repeat digest runs near-instant (~0.02s vs ~70s for 1500 entries).

## Automation

Cron jobs handle daily fetching and weekly digest generation:

```
47 6 * * *   # daily fetch
3  7 * * 1   # weekly digest (Monday)
```

## Project structure

```
rss/
  config.yaml        # interests, tags, email
  feeds.json         # registered feed list
  tasks.py           # invoke entrypoint
  requirements.txt
  src/
    tasks.py         # all task definitions and helpers
  entries/           # fetched entries per feed (JSON)
  cache/             # h-index + embedding caches
  digests/           # generated HTML digests
  logs/              # cron output
```
