# Tribunal evaluation corpus

Machine-readable form of the 56 curated tribunal test cases and the checks that
qualify them for the keyed experiment. Everything here runs offline, without an
API key.

## Artifacts

- `cases.jsonl` — 56 cases (TC01..TC56), one JSON object per line. Each carries
  the article `title`/`text`/`source_url` pulled from its source CSV row, the
  detector `signal`, `control_type`, `search_necessity`, a normalized
  `text_sha256`, provenance (test split vs held-out tail) and the `gold` verdict.
- `manifest.json` — counts, `cases_sha256`, source-CSV hashes, annotation status.
- `check_report.json` — output of the provenance/dedup/leakage checks.
- `ANNOTATION_GUIDE.md` + `annotation/` — the blind sheet and empty template for
  the pending second annotation.

## Regenerate

```
uv run --no-sync python evals/build_corpus.py         # cases.jsonl + manifest
uv run --no-sync python evals/check_corpus.py         # check_report.json
uv run --no-sync python evals/make_annotation_sheet.py
```

Source of truth is `docs/completion-plan/13-curated-corpus.md` (case metadata)
and `14-gold-labels.md` (gold verdicts). The build joins them by case id, checks
that both tables agree on `dataset#row`, and reads the text from the CSV by
integer row position.

## Findings (current run)

- Verdicts: 24 reliable / 23 questionable / 9 unreliable. Signals: jeansa 12,
  clickbait 9, ai 8, mt 19, none 8. 39 of 56 are search-required.
- Integrity and dedup: pass (no empty text, no duplicate cases).
- Test-split: pass — all jeansa and clickbait cases really come from
  `split == 'test'`.
- **Detector leakage (quantified):** of the 32 held-out-tail ai/mt cases, **20
  fall inside the model's reproduced training partition** (ai 7/12, mt 13/20),
  7 in validation, 3+2 in test. So for those items the local detector score fed
  to the tribunal is partly in-sample. This is the tail-leakage caveat made
  numeric; it belongs in the threats-to-validity section. (Membership is
  pre-dedup, an upper bound; split reproduced with seed 42.)
- **Shortcut classifier:** signal -> most-frequent-verdict captures 29/56
  (51.8%) overall and 21/40 (52.5%) on primary items, versus a 3-class chance of
  ~33% and the previous corpus's ~89%. The gold verdict is no longer trivially
  predictable from the detector signal. These figures match 14-gold-labels.md
  exactly, which independently validates this serialization.

## Second annotation + arbitration (done, ML-assisted)

A second **independent, blind** annotation was produced on the **Fable** model
(`annotation/annotator2.fable.jsonl`) and the 17 disagreements were resolved by
Fable arbiters (`annotation/arbitration.jsonl`). The final labels are in
`annotation/gold_final.jsonl` (39 curator+Fable agreements + 17 arbitrated).

- Cohen's kappa (curator vs Fable, pre-arbitration): **0.5145**; full agreement
  **39/56 (69.6%)**.
- Final distribution: reliable 24, questionable 24, unreliable 8. 11 of 56
  differ from the original curator gold.
- **Honesty note:** the second annotation and the arbitration are ML-assisted
  (Fable), NOT human inter-annotator agreement; the thesis must state this, and a
  human can override any label in `gold_final.jsonl`. See
  `annotation/arbitration_report.md`.
