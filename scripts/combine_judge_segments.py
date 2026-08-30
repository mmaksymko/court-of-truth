"""Combine disjoint immutable judge-only segments after verifying provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = _parse_args()
    prompt_out = args.out.parent / f"{args.out.stem}_prompts"
    collisions = [path for path in (args.out, prompt_out) if path.exists()]
    if collisions:
        raise SystemExit(
            "outputs already exist and will not be overwritten: "
            + ", ".join(str(path) for path in collisions)
        )

    rows: list[dict] = []
    prompts: list[Path] = []
    for segment in args.segments:
        rows += _rows(segment)
        prompt = segment.parent / f"{segment.stem}_prompts" / "judge_F.txt"
        if not prompt.exists():
            raise SystemExit(f"missing F prompt snapshot for {segment}: {prompt}")
        prompts.append(prompt)
    keys = [(row["item_id"], row["mode"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit("segments overlap")
    if any(row["mode"] != "F" for row in rows):
        raise SystemExit("only F records are allowed")
    if any(not row.get("label") or not row.get("verdict") for row in rows):
        raise SystemExit("every record must contain a successful full verdict")
    versions = {row["prompt_version"] for row in rows}
    hashes = {row["judge_prompt_sha256"] for row in rows}
    prompt_hashes = {_sha256(path) for path in prompts}
    if len(versions) != 1 or len(hashes) != 1 or hashes != prompt_hashes:
        raise SystemExit("segment prompt provenance does not match")

    rows.sort(key=lambda row: int(row["item_id"][2:]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    prompt_out.mkdir()
    shutil.copy2(prompts[0], prompt_out / "judge_F.txt")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "records": len(rows),
                "prompt_version": next(iter(versions)),
                "judge_prompt_sha256": next(iter(hashes)),
                "runs_sha256": _sha256(args.out),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
