#!/usr/bin/env python3
"""Minimal launcher for bundled CytoFuse examples."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bundled CytoFuse example.")
    parser.add_argument("example", choices=("citeseq", "cytof", "facs"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scripts = {
        "citeseq": ROOT / "cytofuse" / "citeseq" / "run_all_batches.py",
        "cytof": ROOT / "cytofuse" / "cytof" / "run_all_batches.py",
        "facs": ROOT / "cytofuse" / "facs" / "run_all_panels.py",
    }
    output = args.output or (ROOT / "outputs" / f"{args.example}_example")
    cmd = [PYTHON, scripts[args.example], "--output", output]
    if args.force:
        cmd.append("--force")
    cmd.extend(args.extra_args)
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
