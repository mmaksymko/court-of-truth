"""Create an immutable, checksum-indexed snapshot of prompts and run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from court.tribunal.instructions import judge_instructions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("evals/runs"))
    parser.add_argument("--out-root", type=Path, default=Path("evals/run_archive"))
    parser.add_argument("--protocol", type=Path, default=Path("evals/protocol.json"))
    parser.add_argument("--plan", type=Path, default=Path("evals/incremental_run_plan.json"))
    parser.add_argument("--extra", type=Path, nargs="*", default=[])
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = _parse_args()
    destination = args.out_root / args.name
    if destination.exists():
        raise SystemExit(f"archive already exists and will not be overwritten: {destination}")
    destination.mkdir(parents=True)
    shutil.copytree(args.runs_dir, destination / "runs")
    shutil.copy2(args.protocol, destination / "protocol.json")
    shutil.copy2(args.plan, destination / "incremental_run_plan.json")
    if args.extra:
        metadata = destination / "metadata"
        metadata.mkdir()
        for path in args.extra:
            shutil.copy2(path, metadata / path.name)

    prompts = destination / "prompts"
    prompts.mkdir()
    (prompts / "judge_F.txt").write_text(
        judge_instructions(adversarial=True, include_forensics=True), encoding="utf-8"
    )
    (prompts / "judge_B1.txt").write_text(
        judge_instructions(adversarial=False, include_forensics=True), encoding="utf-8"
    )

    files = sorted(path for path in destination.rglob("*") if path.is_file())
    manifest = {
        "archive": args.name,
        "immutable_policy": "archive_run.py refuses to overwrite an existing archive",
        "files": {
            str(path.relative_to(destination)): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"archived {len(files)} files to {destination}")


if __name__ == "__main__":
    main()
