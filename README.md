# Wiki

Personal knowledge base for research, projects, ideas, and business.
Inspired by [karpathy's LLM wiki system](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

## Structure

| Folder | Purpose |
|---|---|
| `sources/` | Raw ingested documents — papers, articles, notes, data |
| `entities/` | Pages about people, organizations, tools, products |
| `concepts/` | Definitions, mental models, technical concepts |
| `projects/` | Active and past projects |
| `business/` | Business ideas, strategy, market research |
| `ideas/` | Raw ideas not yet classified |

## Key Files

- [index.md](index.md) — Content catalog, searchable overview of everything in the wiki
- [log.md](log.md) — Append-only chronological ingest/query/lint log
- [schema.md](schema.md) — Conventions, workflows, and maintenance rules

## Workflows

### Ingest a new source
1. Drop the file into `sources/`
2. Open a conversation with an LLM, paste `schema.md` + the source
3. Ask it to: summarize → create/update entity or concept pages → update `index.md` → append to `log.md`

### Query
1. Open a conversation, load relevant pages from `concepts/` or `entities/`
2. Ask your question — instruct the LLM to cite page names
3. If the answer is worth keeping, save it as a new page

### Lint (periodic health check)
- Ask the LLM to scan for contradictions, orphan pages, stale claims, and missing cross-references
- Log the lint pass in `log.md`
