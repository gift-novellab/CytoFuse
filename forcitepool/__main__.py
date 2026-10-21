"""Unified command-line entry point for ForCitePool."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from . import evaluate, run


COMMANDS: dict[str, tuple[str, Callable[[Sequence[str] | None], None]]] = {
    "run": ("construct a protein-marker taxonomy", run.main),
    "evaluate": ("evaluate a run against optional labels", evaluate.main),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m forcitepool",
        description="Build and assess marker-based CITE-seq cell taxonomies.",
    )
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="arguments passed to the selected command",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    if namespace.command is None:
        parser.print_help()
        print("\ncommands:")
        for name, (description, _) in COMMANDS.items():
            print(f"  {name:<10} {description}")
        return
    COMMANDS[namespace.command][1](namespace.args)


if __name__ == "__main__":
    main()
