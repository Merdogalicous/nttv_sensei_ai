# NTTV Chatbot Tests

This suite protects the structured deterministic stack as well as the ingest and retrieval pipeline.

## What changed

Deterministic extractors no longer return final user-facing prose.
They now return structured `DeterministicResult` payloads, and `nttv_chatbot/composer.py`
is tested separately as the layer that turns those facts into natural-language answers.

## Quick start

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pytest -q
```

Run a focused file:

```powershell
pytest -q tests/test_rank_prompts.py
pytest -q tests/test_composer.py
```

## What these tests cover

- Structured deterministic extractors for rank, schools, weapons, glossary, sanshin, kihon happo, kyusho, leadership, and technique lookups
- Router priority, so the most specific extractor still wins before broader fallbacks
- Deterministic composition, including brief / standard / full output styles
- Chunking, document parsing, and hybrid retrieval behavior

## Conventions

- Prefer deterministic extractors first; unanswered cases should still return `None`
- Extractor unit tests should validate structured facts, `answer_type`, and `det_path`
- Composer tests should validate the final natural-language rendering from those facts
- Keep extractors pure where practical; tests pass passages in directly

## Troubleshooting

- If imports fail, make sure the repo root is on `PYTHONPATH`
- If local data moved, update the small loader helpers in the affected tests
- If a deterministic test fails, check routing first, then fact extraction, then composition
