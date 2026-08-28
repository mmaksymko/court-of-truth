# Second independent annotation — guide

The gold verdicts in `cases.jsonl` were produced by a single curator (the
author). A defensible gold standard needs a **second, independent** annotation
followed by arbitration and an inter-annotator agreement (Cohen's kappa) figure.
This guide and the blind sheet let a second annotator label the 56 cases without
seeing the curator's verdicts, the detector signal, or any system output.

## Blindness rule

Annotate only from `evals/annotation/blind_sheet.jsonl`. Each line has just:
`id`, `title`, `text`, `source_url`, `search_necessity`. It deliberately omits
the detector signal, the control type, the curator verdict and every system
score. Do not open `cases.jsonl`, `14-gold-labels.md` or the detector outputs
before you finish.

## Verdict definitions (assign one per case)

- **reliable** — key factual claims are verified and there are no material
  contradictions.
- **questionable** — key claims are unverified, weakly supported, interdependent,
  or the piece is misleadingly framed. When information is insufficient, choose
  this rather than `unreliable`. Absence of confirmation is not, by itself,
  grounds for `unreliable`.
- **unreliable** — key claims are refuted, fabricated, or systematically
  misleading.

Judge credibility of the content, independent of writing style: AI-generated,
machine-translated, sponsored or clickbait form does not by itself decide the
verdict.

## What to record per case

For each `id` fill one object in your copy of
`evals/annotation/annotator2.template.jsonl`:

- `verdict` — `reliable` | `questionable` | `unreliable`
- `confidence` — 0.50–1.00
- `search_needed` — `true` if you had to check an external fact, else `false`
- `key_claims` — the checkable claim(s) you assessed (short)
- `acceptable_sources` — URLs you used to verify (if any)
- `rationale` — one or two sentences on why

For `search-required` items, verify the concrete claim with your own web search.
For `article-sufficient` items, decide from the text. Never invent a source or a
fact; if something cannot be checked, mark it `questionable` and say so.

## After both annotations exist

Run `uv run --no-sync python evals/agreement.py <annotator2.jsonl>`. It compares
your labels against the curator gold, reports Cohen's kappa and the full-agreement
rate, and writes `evals/annotation/disagreements.json` — the worksheet for a
third-party arbiter to resolve. Arbitrated labels become the frozen gold; the
pre-arbitration kappa is reported in the thesis as the agreement figure.
