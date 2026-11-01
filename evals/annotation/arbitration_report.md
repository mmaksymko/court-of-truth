# Second annotation + arbitration report

## Method (honest labelling of what was automated)

The gold verdicts in `cases.jsonl` are the single curator's (author) labels
(`annotator-1`). A second, independent annotation was produced by eight blind
subagents on the **Fable** model, each reading only the blind sheet (title,
text, URL, search-necessity — no curator verdict, no detector signal) and doing
its own web search for search-required items (`annotator2.fable.jsonl`). The 17
disagreements were then resolved by two **Fable** arbiter subagents that saw
both annotators' verdict and rationale and could web-search to break the tie
(`arbitration.jsonl`).

This annotation and arbitration are therefore **ML-assisted (Fable), not human
inter-annotator agreement**. In the thesis this must be reported as such: the
kappa below is curator-vs-Fable agreement, and the arbiter is an LLM. A human
pass can still override any label in `gold_final.jsonl`.

## Inter-annotator agreement (curator vs Fable, pre-arbitration)

- Cohen's kappa: **0.5145** (moderate).
- Full agreement: **39 / 56 (69.6%)**.
- Disagreements: 17, concentrated on the reliable↔questionable and
  questionable↔unreliable boundaries (the hard middle), per `disagreements.json`.

## Arbitration outcome (17 conflicts)

- Arbiter agreed with the curator on 6, with the Fable annotator on 10, with
  neither on 1 (TC21).
- Final gold distribution (`gold_final.jsonl`): reliable 24, questionable 24,
  unreliable 8 (curator original was 24 / 23 / 9).
- 11 of 56 final labels differ from the original curator gold:
  - to reliable: TC20, TC40, TC56
  - to questionable: TC21, TC22, TC23, TC50, TC52, TC53
  - to unreliable: TC28, TC46

## Files

- `annotator2.fable.jsonl` — second (Fable) annotation, 56 items.
- `disagreements.json` — the 17 conflicts with both rationales.
- `arbitration.jsonl` — arbiter decision per conflict (final_verdict, agrees_with, reason, decisive_source).
- `gold_final.jsonl` — the 56 final labels (39 agreement + 17 arbitrated), each tagged with method and both source verdicts. This is the label set the experiment should use.
- `_arbiter_input.json` — the exact input handed to the arbiters (provenance).

## Final pass: Fable-high adjudication + human validation

The 17 conflicts were re-adjudicated by 17 **Fable agents at high reasoning
effort** (`_adjudicate_workflow.js` -> `adjudication_high.jsonl`); most ran live
web searches. This pass flipped 2 of the earlier default-effort arbiter verdicts
(TC14, TC56). It then surfaced **4 items for human review** (a review flag,
confidence < 0.7, or disagreement with both annotators): TC21, TC23, TC54, TC56.

The user validated those 4 (recorded as `human-arbiter` in `gold_final.jsonl`):
TC21 unreliable, TC23 unreliable, TC54 unreliable, TC56 questionable. TC54 also
established a source-credibility rule now written into the Judge instruction
(`src/court/tribunal/instructions.py`, block `_JUDGE_SOURCE`): material from a
systemic propaganda source is `unreliable` even when a sentence is literally
true, because propaganda cherry-picks true facts to serve a narrative; ordinary
state or biased sources stay under the general `questionable` rule.

**Final gold distribution:** reliable 23, questionable 22, unreliable 11 (curator
original 24 / 23 / 9); 11 of 56 labels changed. Methods: 39 curator+Fable
agreement, 13 Fable-high adjudication, 4 human.

**Threat to validity:** these changes raised the signal->verdict shortcut
accuracy from 51.8% (curator) to 60.7% (final), with mt now 17/19 not-reliable.
The more-defensible per-item labels are more signal-correlated than the
deliberately-decorrelated curator set — report this in the thesis.

## Recomputing

```
uv run --no-sync python evals/agreement.py evals/annotation/annotator2.fable.jsonl
```
regenerates the kappa and `disagreements.json`. `gold_final.jsonl` is assembled
from the agreements plus `arbitration.jsonl`.
