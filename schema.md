# Schema

Conventions and rules for maintaining this wiki.

---

## File Naming

- Lowercase, hyphen-separated: `transformer-architecture.md`, `andrej-karpathy.md`
- No spaces, no special characters

## Frontmatter (YAML)

Every page should start with:

```yaml
---
title: Page Title
type: concept | entity | project | business | idea | source
tags: [tag1, tag2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: [source-filename.md]
---
```

## Wikilinks

Use `[[page-name]]` for cross-references. Always link to the canonical page name (filename without `.md`).

## Page Types

### `concepts/`

- Definition in one paragraph
- Key properties / variants
- Related concepts (wikilinks)
- Sources

### `entities/`

- Who/what it is (one paragraph)
- Why it matters to this wiki
- Key works, products, or contributions
- Related entities and concepts

### `projects/`

- Status: `active | paused | done | abandoned`
- Goal (one sentence)
- Key decisions and why
- Open questions
- Next actions

### `business/`

- Problem being solved
- Target customer
- Rough business model
- Competitive landscape
- Open questions / risks

### `ideas/`

- The raw idea (one paragraph)
- Why it might be valuable
- What would need to be true for it to work
- Next step to validate

---

## Ingest Protocol

When processing a new source, instruct the LLM to:

1. Write a `sources/<filename>.md` summary page
2. Create or update relevant `entities/` and `concepts/` pages
3. Add an entry to `index.md` under the right category
4. Append a line to `log.md`: `## [YYYY-MM-DD] ingest | <Title>`

## Lint Protocol

Periodically ask the LLM to:

- Flag contradictions between pages
- Find orphan pages (nothing links to them)
- Identify stale claims (marked with `[stale?]` tag)
- Check all wikilinks resolve

Log the lint: `## [YYYY-MM-DD] lint | <brief findings>`
