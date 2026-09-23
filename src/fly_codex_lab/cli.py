from __future__ import annotations

import argparse
import json
from pathlib import Path

from .codex_readout import explain_with_codex
from .pipeline import run_pipeline, compare_encoders


def main():
    p = argparse.ArgumentParser(
        description="Text -> semantic encoder -> fly-brain experiment"
    )
    p.add_argument("text")
    p.add_argument("--encoder", choices=["keyword", "random", "codex"], default="keyword")
    p.add_argument("--backend", choices=["proxy", "flybrain"], default="proxy")
    p.add_argument("--seed", type=int, default=5175)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--model", default=None)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--no-codex", action="store_true")
    p.add_argument(
        "--explain",
        action="store_true",
        help="Use Codex only to translate the final experiment JSON into Traditional Chinese.",
    )
    p.add_argument("--debug", action="store_true")
    p.add_argument("--output", default="fly_result.json")
    a = p.parse_args()

    if a.compare:
        data = compare_encoders(
            a.text,
            a.backend,
            a.seed,
            a.steps,
            not a.no_codex,
            a.model,
        )
    else:
        data = run_pipeline(
            a.text,
            a.encoder,
            a.backend,
            a.seed,
            a.steps,
            a.model,
        )

    if a.explain:
        try:
            data["codex_readout_summary"] = explain_with_codex(data)
        except Exception as exc:
            data["codex_readout_error"] = f"{type(exc).__name__}: {exc}"

    Path(a.output).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if a.debug:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"Saved to {a.output}")
        if "codex_readout_summary" in data:
            print("\nCodex readout:")
            print(data["codex_readout_summary"])


if __name__ == "__main__":
    main()
