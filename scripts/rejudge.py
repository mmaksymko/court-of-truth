"""Re-run ONLY the judge over saved party arguments, with a candidate judge prompt.

Cheap calibration experiment: the prosecutor/advocate (F/B2) or neutral (B1)
arguments and the verified evidence are read back from an explicitly selected run file.
The forensic report comes from the saved run/cache; B2 keeps it only as an audit
artifact and never exposes it to the judge. A fresh judge re-scores each case with
no web search, detector, or party rerun. Usage:

    uv run --no-sync python scripts/rejudge.py \
        --runs evals/runs/incremental_v4_combined/runs.jsonl \
        --out evals/runs/rejudge_v4.jsonl --prompt-version v4-central-claim \
        --exclude-ids TC25 TC28 TC31 TC45 TC51
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from court.config import Settings
from court.forensics.schemas import ForensicReport
from court.tribunal import prompts
from court.tribunal.instructions import judge_instructions
from court.tribunal.schemas import Argument, EvidenceRecord, Verdict

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals" / "cases.jsonl"
DEFAULT_FORENSICS = ROOT / "evals" / "runs" / "forensics_existing_v4_input.jsonl"


def _cases() -> dict[str, dict]:
    return {json.loads(line)["id"]: json.loads(line) for line in CASES.read_text().splitlines()}


def _saved(mode: str, runs_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["mode"] == mode and record["status"] == "ok":
            out[record["item_id"]] = record
    return out


def _forensics(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        record["item_id"]: record
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
    }


async def _judge(agent, user: str, runner, run_config):
    result = await runner.run(agent, user, max_turns=2, run_config=run_config)
    return result.final_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--adversarial-prompt", type=Path)
    parser.add_argument("--b2-prompt", type=Path)
    parser.add_argument("--neutral-prompt", type=Path)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--exclude-ids", nargs="+", default=[])
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument(
        "--modes", nargs="+", choices=("F", "B1", "B2", "B3"), default=["F", "B1"]
    )
    parser.add_argument("--forensics-cache", type=Path, default=DEFAULT_FORENSICS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _prompt(path: Path | None, *, adversarial: bool, include_forensics: bool) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    return judge_instructions(adversarial=adversarial, include_forensics=include_forensics)


def _selected_ids(args: argparse.Namespace) -> dict[str, list[str]]:
    requested = set(args.ids) if args.ids else None
    excluded = set(args.exclude_ids)
    selected: dict[str, list[str]] = {}
    for mode in args.modes:
        available = set(_saved(mode, args.runs))
        unknown = (requested or set()) - available
        if unknown:
            raise SystemExit(f"{mode}: no saved successful run for {', '.join(sorted(unknown))}")
        selected[mode] = sorted(
            (available if requested is None else requested) - excluded,
            key=lambda value: int(value[2:]),
        )
    return selected


async def main() -> None:  # noqa: PLR0912, PLR0915
    args = _parse_args()
    prompt_by_mode = {
        "F": _prompt(args.adversarial_prompt, adversarial=True, include_forensics=True),
        "B1": _prompt(args.neutral_prompt, adversarial=False, include_forensics=True),
        "B2": _prompt(args.b2_prompt, adversarial=True, include_forensics=False),
        # B3 = F ablated on web search only: two adversarial parties whose saved
        # arguments carry no search evidence, judged by the SAME prompt as F.
        "B3": _prompt(args.adversarial_prompt, adversarial=True, include_forensics=True),
    }
    out_path = args.out
    selected = _selected_ids(args)
    cases = _cases()
    cached_forensics = _forensics(args.forensics_cache)
    saved_by_mode = {mode: _saved(mode, args.runs) for mode in args.modes}
    missing_reports = sorted(
        {
            item_id
            for mode, ids in selected.items()
            for item_id in ids
            if not saved_by_mode[mode][item_id].get("forensic_report")
            and item_id not in cached_forensics
        }
    )
    if missing_reports:
        raise SystemExit(f"missing cached forensic reports: {', '.join(missing_reports)}")
    selected_item_ids = {item_id for ids in selected.values() for item_id in ids}
    for item_id, cached in cached_forensics.items():
        if (
            item_id in selected_item_ids
            and cached["case_text_sha256"] != cases[item_id]["text_sha256"]
        ):
            raise SystemExit(f"{item_id}: cached forensic report does not match article text")
    if args.dry_run:
        total = sum(len(ids) for ids in selected.values())
        print(f"DRY RUN: {total} judge-only calls; no party or web-search reruns")
        for mode, ids in selected.items():
            print(f"{mode}: {len(ids)} items ({' '.join(ids)})")
        return

    prompt_dir = out_path.parent / f"{out_path.stem}_prompts"
    collisions = [path for path in (out_path, prompt_dir) if path.exists()]
    if collisions:
        raise SystemExit(
            "outputs already exist and will not be overwritten: "
            + ", ".join(str(path) for path in collisions)
        )

    from agents import (  # noqa: PLC0415
        Agent,
        ModelSettings,
        OpenAIResponsesModel,
        RunConfig,
        Runner,
    )
    from openai import AsyncOpenAI  # noqa: PLC0415
    from openai.types.shared import Reasoning  # noqa: PLC0415

    settings = Settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    model = OpenAIResponsesModel(model=settings.model, openai_client=client)
    msettings = ModelSettings(store=False, reasoning=Reasoning(effort="medium"))
    judge_names = {"F": "СуддяF", "B1": "ДослідникСуддяB1", "B2": "СуддяB2", "B3": "СуддяB3"}
    judges = {
        mode: Agent(
            name=judge_names[mode],
            instructions=prompt_by_mode[mode],
            model=model,
            model_settings=msettings,
            output_type=Verdict,
        )
        for mode in selected
    }
    run_config = RunConfig(tracing_disabled=True)
    prompt_hashes = {
        mode: hashlib.sha256(prompt_by_mode[mode].encode()).hexdigest() for mode in selected
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir()
    for mode in selected:
        (prompt_dir / f"judge_{mode}.txt").write_text(prompt_by_mode[mode], encoding="utf-8")
    handle = out_path.open("w", encoding="utf-8")
    for mode in args.modes:
        saved = saved_by_mode[mode]
        for item_id in selected[mode]:
            record = saved[item_id]
            case = cases[item_id]
            report_data = record.get("forensic_report")
            if report_data is None:
                report_data = cached_forensics[item_id]["report"]
            report = ForensicReport.model_validate(report_data)
            arguments = [Argument.model_validate(a) for a in record["arguments"]]
            evidence = [EvidenceRecord.model_validate(e) for e in record["evidence"]]
            if mode in {"F", "B2", "B3"}:
                presented = [("prosecution", arguments[0]), ("defence", arguments[1])]
            else:
                presented = [("research", arguments[0])]
            user = prompts.judge(
                case["title"],
                case["text"],
                report,
                presented,
                evidence,
                source_url=case.get("source_url"),
                include_forensics=mode != "B2",
                published=case.get("published"),
            )
            try:
                verdict: Verdict = await _judge(judges[mode], user, Runner, run_config)
                handle.write(
                    json.dumps(
                        {
                            "item_id": item_id,
                            "mode": mode,
                            "label": verdict.label,
                            "probabilities": verdict.probabilities.as_map(),
                            "verdict": verdict.model_dump(mode="json"),
                            "run_kind": "judge-only",
                            "prompt_version": args.prompt_version,
                            "judge_prompt_sha256": prompt_hashes[mode],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
                print(f"{mode} {item_id} -> {verdict.label}", flush=True)
            except Exception as exc:
                handle.write(
                    json.dumps(
                        {
                            "item_id": item_id,
                            "mode": mode,
                            "label": None,
                            "error": str(exc)[:200],
                            "run_kind": "judge-only",
                            "prompt_version": args.prompt_version,
                            "judge_prompt_sha256": prompt_hashes[mode],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
                print(f"{mode} {item_id} -> FAIL {str(exc)[:80]}", flush=True)
    handle.close()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
