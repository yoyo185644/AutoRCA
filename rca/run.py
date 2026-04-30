from __future__ import annotations

import argparse
from typing import Callable, List


def _runner_entrypoints() -> dict[str, Callable[[List[str] | None], int]]:
    from rca.run_agent_standard import cli_main as agent_standard_main

    return {
        "agent_standard": agent_standard_main,
    }


def cli_main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=str, default="agent_standard")
    known, remaining = parser.parse_known_args(argv)

    eps = _runner_entrypoints()
    name = str(known.runner or "").strip()
    if name not in eps:
        raise SystemExit(f"Unknown runner: {name}. Available: {', '.join(sorted(eps.keys()))}")
    return int(eps[name](remaining))


if __name__ == "__main__":
    raise SystemExit(cli_main())
