# NTTV Chatbot - Deterministic RAG Assistant for Ninja Training TV

A RAG-based, extractor-driven chatbot for Ninja Training TV (NTTV).

The stack centers on:
- deterministic extractors for known-known questions
- FAISS retrieval over embedded curriculum content
- OpenRouter-compatible LLM generation
- a local-first ingest pipeline that builds `index/` from `data/`

## Key Features

- Deterministic extractors for rank requirements, kihon, sanshin, schools, weapons, kyusho, and related topics.
- FAISS `IndexFlatIP` retrieval over normalized embeddings from `sentence-transformers/all-MiniLM-L6-v2`.
- OpenRouter-compatible model routing through environment variables.
- Shared ingest pipeline for `.txt`, `.md`, `.docx`, and now structured `.pdf` parsing.
- Optional Docling-backed PDF ingestion with page-aware and heading-aware metadata.
- Section-aware, metadata-rich chunking with stable chunk IDs and token-based controls.
- Hybrid retrieval with FAISS dense search, BM25 lexical search, RRF fusion, and optional reranking.

## Architecture

### Ingestion

`ingest.py`:
- scans `data/`
- parses files through `nttv_chatbot/document_parsing.py`
- chunks parsed output through `nttv_chatbot/chunking.py`
- embeds chunks
- writes:
  - `index/index.faiss`
  - `index/faiss.index`
  - `index/meta.pkl`
  - `index/config.json`

Text files keep the existing behavior through the new parser interface:
- `.txt`
- `.md`
- `.docx`

PDF files:
- use Docling as the primary parser when available
- can fall back cleanly to `pypdf` when `PDF_FAIL_OPEN=true`
- never stop the whole ingest run because one PDF failed

Chunking:
- groups structured material by section first
- keeps headings attached to their content
- prefers paragraph boundaries over blind splitting
- supports token-based target/max/min/overlap settings
- merges tiny fragments into better retrieval chunks when possible
- preserves stable source traceability in every chunk

### Retrieval

`app.py` + `nttv_chatbot/retrieval.py`:
- preserves deterministic extractor short-circuits before retrieval for known-known questions
- lazily loads FAISS plus `meta.pkl`
- runs dense FAISS retrieval and lexical BM25 retrieval in parallel stages
- fuses dense + lexical candidates with Reciprocal Rank Fusion (RRF)
- applies priority-aware heuristics as a ranking stage
- optionally reranks the fused shortlist with Jina and falls back safely to heuristic ordering
- uses OpenRouter-compatible LLM calls for synthesis

## Repository Structure

```text
nttv_chatbot_ext/
|-- app.py
|-- api_server.py
|-- ingest.py
|-- extractors/
|-- data/
|-- index/
|-- nttv_chatbot/
|   |-- config.py
|   |-- chunking.py
|   |-- document_parsing.py
|   |-- retrieval.py
|   `-- llm_client.py
|-- tests/
|-- requirements.txt
|-- render.yaml
`-- README.md
```

## Installation

### 1) Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install the base dependencies

```bash
python -m pip install -U pip
pip install -r requirements.txt
```

### 3) Optional: install Docling for structured PDF ingestion

Docling is intentionally optional. Base text ingestion does not require it.

```bash
pip install docling==2.88.0
```

If you deploy to Render or another host and want PDF ingestion there too, install Docling in that environment as well.

## Environment Variables

### Core app/runtime

| Variable | Example | Purpose |
|---|---|---|
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter/OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | `sk-or-...` | API key |
| `MODEL` | `google/gemma-3n-e4b-it` | Primary synthesis model |
| `INDEX_DIR` | `index` | Index directory root |
| `INDEX_PATH` | `index/faiss.index` | FAISS index path |
| `META_PATH` | `index/meta.pkl` | Chunk metadata path |
| `TOP_K` | `6` | Retrieval depth |
| `TEMPERATURE` | `0.0` | Generation temperature |
| `MAX_TOKENS` | `512` | Generation cap |

### PDF ingest

| Variable | Example | Purpose |
|---|---|---|
| `ENABLE_PDF_INGEST` | `true` | Enables PDF ingest. Default behavior is automatic enablement when Docling is installed. |
| `PDF_PARSER` | `docling` | PDF parser selection. Supported values: `docling`, `pypdf`. |
| `PDF_PARSE_MAX_PAGES` | `25` | Optional max pages per PDF during ingest. |
| `PDF_FAIL_OPEN` | `true` | If Docling fails, fall back cleanly or skip the PDF instead of killing ingest. |

### Chunking

| Variable | Example | Purpose |
|---|---|---|
| `CHUNK_TARGET_TOKENS` | `180` | Soft target size for section-aware chunks. |
| `CHUNK_MAX_TOKENS` | `240` | Hard cap before chunk splitting. |
| `CHUNK_OVERLAP_TOKENS` | `40` | Overlap budget carried into the next chunk. |
| `CHUNK_MIN_TOKENS` | `60` | Small-fragment merge threshold. |

### Retrieval

| Variable | Example | Purpose |
|---|---|---|
| `USE_HYBRID_RETRIEVAL` | `true` | Enables dense + lexical retrieval together. |
| `DENSE_TOP_K` | `12` | Dense FAISS candidate count before fusion. |
| `LEXICAL_TOP_K` | `12` | Lexical BM25 candidate count before fusion. |
| `FUSED_TOP_K` | `10` | Fused shortlist size before final context assembly. |
| `RERANKER_BACKEND` | `none` | Supported values: `none`, `heuristic_only`, `jina_api`. |
| `JINA_API_KEY` | `` | Optional Jina reranker API key. |

### Example `.env`

```env
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-xxxx
MODEL=google/gemma-3n-e4b-it

INDEX_DIR=index
INDEX_PATH=index/faiss.index
META_PATH=index/meta.pkl

ENABLE_PDF_INGEST=true
PDF_PARSER=docling
PDF_PARSE_MAX_PAGES=
PDF_FAIL_OPEN=true

CHUNK_TARGET_TOKENS=180
CHUNK_MAX_TOKENS=240
CHUNK_OVERLAP_TOKENS=40
CHUNK_MIN_TOKENS=60

USE_HYBRID_RETRIEVAL=true
DENSE_TOP_K=12
LEXICAL_TOP_K=12
FUSED_TOP_K=10
RERANKER_BACKEND=none
JINA_API_KEY=

TOP_K=6
TEMPERATURE=0.0
MAX_TOKENS=512
STREAMLIT_BROWSER_GATHERUSAGESTATS=false
```

## Build the Index

```bash
python ingest.py
```

The ingest run now prints:
- parser settings
- chunk settings
- per-file parser used
- per-file element counts
- skipped file reasons

### Output Metadata

Each chunk in `index/meta.pkl` still preserves the existing keys used by the app:
- `text`
- `source`
- `meta.priority`
- `meta.source`

Structured ingest adds these metadata fields when available:
- `chunk_id`
- `meta.parser`
- `meta.source_file`
- `meta.file_type`
- `meta.page`
- `meta.page_start`
- `meta.page_end`
- `meta.heading_path`
- `meta.section_title`
- `meta.content_type`
- `meta.rank_tag`
- `meta.school_tag`
- `meta.weapon_tag`
- `meta.technique_tag`
- `meta.priority_bucket`
- `meta.char_count`
- `meta.estimated_token_count`
- `meta.raw_metadata`

`index/config.json` now also records:
- `files`
- `skipped_files`
- `parsing`
- `chunking`

### Retrieval Strategy

The retrieval stack is now stage-based and inspectable:
- deterministic extractors get the first chance to answer before retrieval runs
- dense retrieval uses FAISS over the existing embedding index
- lexical retrieval uses BM25 when `rank-bm25` is installed, with a safe local token-overlap fallback
- fusion uses Reciprocal Rank Fusion so dense-only hits and lexical-only hits can both survive
- heuristics remain in the pipeline as a ranking stage instead of being the whole retriever
- optional Jina reranking only touches the small fused shortlist and falls back safely if config or the API is unavailable

Debug mode in `app.py` now shows:
- dense candidates
- lexical candidates
- fused candidates
- reranked candidates
- stage scores when available

### Chunking Strategy

The chunk builder is section-aware first and size-aware second:
- rank requirements, technique descriptions, and sectioned PDF output are chunked by heading path before token limits apply
- flat text falls back to paragraph-aware chunking
- short sections stay intact when possible
- larger sections split on paragraph boundaries with overlap carried forward for retrieval continuity
- chunk IDs are stable for the same source content, which makes re-ingests easier to reason about

## PDF Support

Place PDFs in `data/` and run:

```bash
python ingest.py
```

Behavior:
- If Docling is installed and PDF ingest is enabled, PDFs are parsed structurally.
- If `PDF_FAIL_OPEN=true` and Docling fails for a PDF, ingest falls back to `pypdf` when possible.
- If a PDF still cannot be parsed, that file is skipped and the rest of ingest continues.
- `.txt` and `.md` ingestion remains unchanged in behavior, just routed through the shared parser interface.

## Run the App

### Streamlit

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

### FastAPI

```bash
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

## Deploying to Render

Base build:

```bash
python -m pip install -U pip && pip install -r requirements.txt && python ingest.py
```

If you want structured PDF ingestion on Render too, make sure Docling is installed in that build environment before `python ingest.py` runs.

Suggested start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Troubleshooting

### Index/meta mismatch

- Rebuild with `python ingest.py`
- Make sure `index.faiss` and `meta.pkl` came from the same ingest run

### PDF files are being skipped

- Install Docling with `pip install docling==2.88.0`
- Or set `ENABLE_PDF_INGEST=true` and `PDF_PARSER=pypdf`
- Check the skipped-file reasons printed by `ingest.py`

### Docling is installed but ingest is slow

- Try `PDF_PARSE_MAX_PAGES=25` for a bounded first pass
- Use `PDF_PARSER=pypdf` if you want a lighter-weight fallback-only mode

### LLM errors

- Check `OPENAI_API_KEY`
- Check `OPENAI_BASE_URL`
- Check `MODEL`

## Tests

Run the full suite:

```bash
pytest
```

Run just the parsing tests:

```bash
pytest tests/test_document_parsing.py
```

Run the chunking tests:

```bash
pytest tests/test_chunking.py
```

Run the retrieval tests:

```bash
pytest tests/test_retrieval.py
```

## License

MIT
