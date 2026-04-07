# Lint Prompt

Use this prompt for periodic wiki health checks. Load all pages from `concepts/`, `entities/`, `sources/`, `index.md`, and `log.md`, then run the following audit:

---

Find and report:

1. **Contradictions** — claims in one page that conflict with claims in another. Cite both pages and the conflicting quotes.

2. **Stale claims** — facts that newer sources or the current date may have superseded (version numbers, "rising" technologies that have since landed or failed, unresolved TODOs). Mark with `[stale?]`.

3. **Orphan pages** — pages that no other wiki page links to via `[[wikilink]]` or markdown link.

4. **Missing concept/entity pages** — `[[wikilinks]]` that appear in page content but have no corresponding `.md` file.

5. **Broken wikilink naming** — links whose slug doesn't match any canonical filename (e.g. `[[mfu]]` when the file is `1-overview-mfu-model-flops-utilization.md`).

6. **Missing cross-references** — two pages that are clearly related but neither links to the other.

7. **Data gaps** — important topics mentioned in passing that deserve their own page or a web search to fill out.

8. **Index gaps** — entries in `concepts/` or `entities/` that are missing from `index.md`.

For every finding, cite the specific file(s) and the relevant line or quote. Be exhaustive — this is a lint pass, not a summary. After the report, append a lint entry to `log.md`:

```text
## [YYYY-MM-DD] lint | <one-line summary of key findings>
```
