"""Run the eval suite against a real model.

    export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b-instruct   # or any OpenAI-compatible API
    python evals/run_evals.py --k 3

Prints a table (passes per scenario and pass^k) and writes a JSON report to evals/results/. With
--min-pass-rate the exit code is non-zero below the threshold, so the suite can gate a CI job.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import anyio

from libassist.agent.llm import LLMError, OpenAICompatibleChatModel
from libassist.evaluation.harness import format_report, load_scenarios, run_suite

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=3, help="runs per scenario (pass^k)")
    parser.add_argument("--scenarios", type=Path, default=HERE / "scenarios.yaml")
    parser.add_argument("--only", nargs="*", help="scenario ids to run")
    parser.add_argument("--min-pass-rate", type=float, default=None, help="e.g. 0.9; fail below it")
    args = parser.parse_args()

    try:
        model = OpenAICompatibleChatModel.from_env()
    except LLMError as exc:
        sys.exit(str(exc))
    scenarios = load_scenarios(args.scenarios)
    if args.only:
        scenarios = [s for s in scenarios if s.id in set(args.only)]

    print(f"Running {len(scenarios)} scenarios x {args.k} with {model.name}...\n")
    reports = anyio.run(partial(run_suite, scenarios, lambda _scenario: model, k=args.k))
    print(format_report(reports, args.k))

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{model.model.replace('/', '_').replace(':', '_')}.json"
    out_file.write_text(
        json.dumps(
            {
                "model": model.name,
                "k": args.k,
                "scenarios": [
                    {
                        "id": r.scenario.id,
                        "passes": r.passes,
                        "pass_all": r.pass_all,
                        "runs": [asdict(run) for run in r.runs],
                    }
                    for r in reports
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Report written to {out_file.relative_to(HERE.parent)}")

    rate = sum(r.pass_all for r in reports) / max(1, len(reports))
    if args.min_pass_rate is not None and rate < args.min_pass_rate:
        sys.exit(f"pass^{args.k} rate {rate:.0%} is below {args.min_pass_rate:.0%}")


if __name__ == "__main__":
    main()
