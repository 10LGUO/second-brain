# Log

Append-only chronological record of all wiki activity.

Format: `## [YYYY-MM-DD] <operation> | <title/description>`
Operations: `ingest`, `query`, `lint`, `create`, `update`
Tag `[Code Change]` for changes to wiki infrastructure (ingest.py, schema.md, .markdownlint.json, lint.md) rather than knowledge content.

---

## [2026-04-05] create | Wiki initialized

## [2026-04-05] [Code Change] create | ingest.py — chunked PDF ingestion pipeline with OCR fallback (pdfplumber + Tesseract chi_sim+eng)

## [2026-04-05] ingest | 1. General Overview — OCR fallback used (scanned image PDF)

## [2026-04-05] ingest | 2. PyTorch Detailed Explanation

## [2026-04-05] ingest | 1. General Overview — re-ingested with higher token limits and English output

## [2026-04-05] update | Translated all Chinese/pinyin content to English across wiki; renamed zongti-jieshao → overview in all filenames and content

## [2026-04-05] [Code Change] update | ingest.py — OCR auto-detection, English-only output instruction, increased token limits (8K/16K)

## [2026-04-06] ingest | 5. Kernel Dev — Operator Development (scanned PDF, OCR used)

## [2026-04-06] [Code Change] create | .markdownlint.json — lint config (MD013, MD025, MD041 disabled)

## [2026-04-06] lint | Fixed markdown lint across all pages: frontmatter ```yaml fences, MD040/MD060/MD031/MD032/MD047

## [2026-04-06] lint | Audit: duplicates, orphans, broken wikilinks, missing pages, index gaps — see findings in lint.md

## [2026-04-06] [Code Change] create | lint.md — reusable wiki health-check prompt

## [2026-04-07] query | Why customize CUDA kernels? — balance point, arithmetic intensity, warp mechanics, vectorized access

## [2026-04-07] update | Expanded concepts/5-kernel-dev-arithmetic-intensity.md — balance point interpretation, A100 example, optimization strategy table

## [2026-04-07] create | concepts/warp.md — warp definition, SIMT relationship, memory coalescing, float4 vs float8 instruction width limit

## [2026-04-07] update | concepts/5-kernel-dev-simt-programming-model.md — added Warp section, fixed frontmatter

## [2026-04-07] lint | Completed 12 truncated concept/entity/source pages; deleted 18 empty stubs and superseded duplicates; fixed all remaining lint errors
