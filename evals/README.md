# Tribunal evaluation corpus

Machine-readable form of the 60 curated tribunal test cases, the human gold
labels, the integrity checks, and the completed ablation run. Everything under
`build_corpus.py` / `check_corpus.py` runs offline, without an API key; the run
under `runs/` was produced against the keyed model.

## Artifacts

- `cases.jsonl` — 60 cases (TC01..TC60), one JSON object per line. Each carries
  the article `title`/`text`/`source_url` pulled from its source CSV row, the
  detector `signal`, `control_type`, `search_necessity`, a normalized
  `text_sha256`, provenance (`split`, `leakage_caveat`) and a legacy `gold`
  verdict (see below). Reproducible from the plan tables (`build_corpus.py`
  regenerates it byte-for-byte).
- `annotation/human_gold.jsonl` — **the canonical, currently pending-rebalance
  gold** (`{id, final_verdict}`, one line per case). The earlier independent curator
  revision `gold-rebalance-20-20-20-2026-08-30` fixed the balance at 20/20/20;
  the later user clarification changed five labels to an interim 17/21/22 and
  marked TC06, TC16 and TC37 for replacement. Do not launch the next experiment
  until five clarity replacements restore exact 20/20/20;
  `annotation/human_gold_history.jsonl` keeps the append-only rationale,
  timestamp, source URL and prior decisions.
  `cases.jsonl.gold.verdict` is an **earlier plan-table label kept for audit
  only** — it differs from the human gold on 15 of 60 cases and is never scored
  against. The older `annotation/gold_final.jsonl` (single curator + Fable
  ML-arbitration) is **superseded** by the human annotation and retained only for
  audit trail.
- `manifest.json` — `human_gold_verdict_counts` (canonical), `legacy_verdict_counts`
  (audit), `human_gold_annotator_counts`, `signal_counts`, `cases_sha256`,
  `human_gold_sha256`, source-CSV hashes, annotation status.
- `protocol.json` — run descriptor; `gold_labels_path` points at
  `annotation/human_gold.jsonl`, `gold_labels_key = final_verdict`, and status is
  `pending-gold-rebalance-v7-prompt-ready-not-run`. `incremental_run_plan.json` pins the
  exact five-case full run and the 55-case F/B1 judge-only refresh; the protocol's
  `judge_v6_follow_up` block pins the later F-only calibration/holdout run.
- `check_report.json` — integrity (real CSV re-read + hash), dedup + near-duplicate
  pairs, test-split, quantified detector leakage, and shortcut baselines
  (signal / domain / signal+domain against the human gold).
- `runs/` — `runs.jsonl` (240 records = 4 modes x 60 cases, one deliberation each),
  `search_transcript.jsonl` (frozen web-search evidence pool).
- `incremental_run_plan.json` — dry-run-verified commands for the v4 incremental
  refresh and the completed output hashes.
- `runs/forensics_existing_v4_input.jsonl` — 55 locally reconstructed and frozen
  deterministic forensic reports for unchanged articles. The historical run did
  not persist these objects; the paid judge-only refresh reads this cache and
  does not rerun detectors or parties.

Every newly produced `RunRecord` now persists its full `forensic_report` in
`runs.jsonl`, including failed tribunal runs. Future judge-only refreshes can
reuse the exact saved report directly; the separate reconstruction cache is
needed only for historical records created before this field existed.

## Completed v4 incremental refresh

The requested scoped refresh completed with **240/240 successful combined
records**:

- five replacement articles ran end-to-end in F/B1/B2/B3 (20 records);
- the other 55 articles reran only the F/B1 judge over saved arguments,
  evidence and frozen forensic reports (110 judge-only records);
- B2/B3 for those 55 unchanged articles remain the historical records, as
  requested.

All-60 v4-comparable results are therefore F and B1:

| mode | accuracy | macro-F1 | Brier | failures |
|------|----------|----------|-------|----------|
| F | 0.683 | 0.690 | 0.410 | 0 |
| B1 | 0.700 | 0.697 | 0.449 | 0 |

The combined artifact also reports B2 accuracy 0.667 / macro-F1 0.664 and B3
accuracy 0.333 / macro-F1 0.167, but those two are deliberately hybrid metrics:
55 historical pre-v4 verdicts plus five new v4 runs. Do not describe them as a
full-corpus v4 rerun.

Artifacts: `runs/incremental_v4_new_cases/`,
`runs/rejudge_v4_existing.jsonl`, and `runs/incremental_v4_combined/`.

## Completed v6 F judge refresh

The F judge was updated with a generic adversarial evidence-synthesis and
materiality gate. It contains no case IDs, domains, articles or gold-label-specific
rules. Prosecutor/advocate outputs, forensic reports, evidence and web-search
results were frozen: only the F judge was called again. B1 remained byte-identical
to v4 (SHA-256 `f5c125...`).

The preregistered, class-balanced split used 24 calibration cases and a 36-case
holdout. v5 failed its calibration gate and was archived without opening holdout.
v6 passed, was frozen (SHA-256 `6de515...`), and the holdout was opened once.

| scope / mode | accuracy | macro-F1 | Brier |
|--------------|----------|----------|-------|
| holdout F v6 | 0.750 | 0.757 | 0.319 |
| holdout B1 v4 | 0.694 | 0.698 | 0.446 |
| all-60 F v6 | 0.733 | 0.740 | 0.363 |
| all-60 F v4 | 0.683 | 0.690 | 0.410 |
| all-60 B1 v4 | 0.700 | 0.697 | 0.449 |

The holdout point estimate for F v6 exceeds B1 by 5.6 percentage points in
accuracy and 0.059 macro-F1. The paired uncertainty remains wide: accuracy 95%
bootstrap CI `[-0.083, 0.194]`, exact McNemar `p=0.6875`. This is a successful
anti-overfit holdout direction, not proof of a stable population-level advantage.

Primary artifacts: `judge_v6_preregistration.json`,
`runs/rejudge_v6_calibration.jsonl`, `runs/rejudge_v6_holdout.jsonl`,
`runs/rejudge_v6_full60.jsonl`, their adjacent prompt snapshots and metric files.
Immutable checkpoints live under `run_archive/`; archive creation refuses to
overwrite an existing checkpoint.

## Regenerate

```
uv run --no-sync python evals/build_corpus.py         # cases.jsonl + manifest
uv run --no-sync python evals/check_corpus.py         # check_report.json
```

Source of truth is `docs/completion-plan/13-curated-corpus.md` (case metadata)
and `14-gold-labels.md` (legacy label). The build joins them by case id, checks
that both tables agree on `dataset#row`, and reads the text from the CSV by
integer row position. The **canonical labels live in `annotation/human_gold.jsonl`**,
not in those plan tables.

## Corpus composition (N = 60)

- **Canonical pending-rebalance gold:** 17 reliable / 21 questionable / 22 unreliable.
  The archived v6 metrics were scored against the prior 20/20/20 hash and must
  not be presented as metrics on this interim revision.
- **Detector signals:** mt 21, none 21, ai 12, jeansa 6. (No `clickbait` case
  survived into the final corpus; the clickbait detector is therefore not
  exercised by this eval set.)
- `search_necessity`: 50 search-required / 10 article-sufficient. Every case now
  has a `source_url` (0 missing).

## Integrity findings (`check_report.json`)

- **Integrity + dedup: pass.** Integrity re-reads each source CSV row and compares
  its hash (not just previously-stored hashes). No empty text, no exact
  duplicates, **no near-duplicate pairs** (token-Jaccard threshold 0.3).
- **Test-split: pass** — the split-bearing datasets (jeansa/clickbait) come from
  `split == 'test'`.
- **Detector leakage (quantified, threats-to-validity):** `ai_generated` and
  `mt_translation` have **no split field** (only jeansa/clickbait do), so their
  "held-out tail" is held out from the jeansa/clickbait split, not from their own
  detector. Of the 53 tail (ai+mt) cases, **32 fall inside the model's reproduced
  training partition** (seed 42) — i.e. ~53% of the corpus is in detector train.
  For those items the local detector score fed to the tribunal is partly
  in-sample. (Membership is pre-dedup, an upper bound.)
- **Shortcut baselines remain high.** Most-frequent-verdict accuracy against the
  current interim gold (all 60): **signal 0.717, domain 0.783, signal+domain 0.783**
  (primary-only: 0.712 / 0.788 / 0.788). The corpus is still
  disinformation-heavy: 21 of 60 cases are `ua.news-pravda.com` (Pravda /
  Portal Kombat network), 15 of the 20 `unreliable` cases carry the `mt` signal,
  and the `mt` signal set is **exactly collinear** with that domain (21 == 21)
  and with the Russian-disinformation topic. Detector signal, machine-translation,
  domain and topic cannot be separated on this corpus — an inherent property of
  the material, not a fixable flaw. The domain shortcut (0.767) is an in-sample
  oracle (it peeks at the gold to pick each group's majority) and is **not a
  deployable classifier**, but it must be reported next to the system's accuracy
  predictor. Existing system metrics predate the gold revision and must not be
  compared with this new baseline until a clean rerun.

## Historical pre-revision ablation results (`runs/runs.jsonl`, N = 60)

The values below were scored against the previous gold hash
`9b0b4688c426368e52909df6bf8748868646ddb3cf2071976419f5ae97dbcc7f`.
They are retained for audit only and are not post-revision results.

Four configurations, each removing one pillar of the full system **F** (two
adversarial parties + forensic detectors + web search):

| mode | ablation | accuracy [95% CI] | macro-F1 [95% CI] | Brier |
|------|----------|-------------------|-------------------|-------|
| **F**  | none (full)        | 0.783 [0.68, 0.88] | **0.791** [0.68, 0.89] | 0.316 |
| **B1** | no adversarial split | 0.750 [0.63, 0.85] | 0.765 [0.64, 0.86] | 0.361 |
| **B2** | no forensic detectors | 0.700 [0.58, 0.82] | 0.687 [0.55, 0.81] | 0.424 |
| **B3** | no web search      | 0.350 [0.23, 0.47] | 0.193 [0.13, 0.26] | 0.907 |

Per-pillar contribution (macro-F1 difference, paired bootstrap, 3000 resamples):

- **web search** `F - B3 = +0.598` [0.480, 0.698] — CI excludes 0, decisive.
- **forensic detectors** `F - B2 = +0.104` [0.001, 0.222] — CI barely excludes 0,
  marginally positive.
- **adversarial split** `F - B1 = +0.026` [-0.085, 0.139] — CI includes 0, **no
  significant benefit** over a single neutral researcher.

Bootstrap baselines vs the human gold: majority-class 0.450, signal-shortcut
0.800, domain-shortcut 0.833. All 240 runs completed (0 final technical failures;
transient 400/rate-limit errors were retried).

**Reading:** search is the load-bearing component; without it the system collapses
(B3 predicts 0/13 reliable and macro-F1 falls to 0.19). Detectors help modestly.
The adversarial prosecutor/advocate split is not measurably better than one
balanced neutral researcher on this corpus. At N=60 the confidence intervals are
wide (±0.10-0.12 on accuracy), so these are directional findings, not tight
estimates.

## Field legend (control_type, dataset_label)

`control_type` (why a case is in the corpus): `primary` (52) ordinary case;
`fabricated-bait` (5) fabricated claim probe; `style-decoy` (3) heavy style signal
with benign content.

`dataset_label` (raw source label): `ai_translated_ua` (21, = mt), `human_ua`
(14), `ai_generated` (12), `human_news` (6), `sponsored` (6, jeansa),
`editorial` (1).
These are provenance tags and do **not** map one-to-one onto the 3-class gold.

## Threats to validity (summary)

1. **No human IAA.** The latest gold is an independent curator revision with
   written rationales for all 60 cases, but there is no second human annotation;
   no human inter-annotator-agreement figure is claimed.
2. **Domain/topic confound.** `mt` signal ≡ `ua.news-pravda.com` domain ≡ Russian
   disinformation topic (21 == 21); contributions of the mt detector, of
   translation, and of the domain prior cannot be disentangled.
3. **Detector leakage.** ~50% of cases are in the ai/mt detector training
   partition, so detector scores fed to the tribunal are partly in-sample.
4. **Small N and pending balance.** N=60 is temporarily 17/21/22 while five
   clarity replacements are pending; the target remains exact 20/20/20. Confidence
   intervals remain wide. Recompute all metrics after the replacements.
5. **Superseded artifacts.** `gold_final.jsonl` and the Fable second-annotation /
   arbitration files are earlier ML-assisted stages, kept for audit only; the
   canonical gold is the human annotation in `human_gold.jsonl`.
