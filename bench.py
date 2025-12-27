import argparse
import asyncio
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from fandom_bench.dataset import BenchmarkRecord, load_dataset
from fandom_bench.providers import ProviderConfig
from fandom_bench.runner import BenchmarkResult, BenchmarkRunner
from fandom_bench.supervisor import SupervisorConfig


def parse_provider_arg(raw: str) -> ProviderConfig:
    parts = raw.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("Provider must look like provider:model[:name]")
    provider, model = parts[0], parts[1]
    name = parts[2] if len(parts) > 2 else None
    return ProviderConfig(provider=provider, model=model, name=name)


def render_results(dataset: List[BenchmarkRecord], results: List[BenchmarkResult]) -> None:
    print("\n=== Provider Status ===")
    by_provider = {}
    for res in results:
        by_provider.setdefault(res.provider, res.status)
    for provider, status in by_provider.items():
        print(f"{provider}: {status}")

    print("\n=== Results ===")
    for res in results:
        sup = ""
        if res.supervisor:
            sup = f"score={res.supervisor.score:.2f} decision={res.supervisor.decision}"
        latency = f" latency={res.latency_seconds:.2f}s" if res.latency_seconds is not None else ""
        print(f"[{res.record_id}] {res.provider}: {res.status}{latency}")
        if res.detail:
            print(f"  detail: {res.detail}")
        if res.output:
            print(f"  output: {res.output}")
        if sup:
            print(f"  supervisor: {sup}")

    print(f"\nProcessed {len(dataset)} records across {len(by_provider)} providers.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV-driven async LLM benchmarker.")
    parser.add_argument("--csv", required=True, help="Path to CSV file.")
    parser.add_argument("--provider", action="append", dest="providers", type=parse_provider_arg, required=True)
    parser.add_argument("--supervisor-model", default="gpt-5", help="OpenAI model used for grading.")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--expected-column", default="expected_answer")
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="Skip network calls; only report wiring.")
    parser.add_argument("--no-supervisor", action="store_true", help="Disable grading.")

    args = parser.parse_args()
    load_dotenv()
    csv_path = Path(args.csv)
    dataset = load_dataset(
        csv_path,
        prompt_column=args.prompt_column,
        expected_column=args.expected_column,
        max_records=args.max_records,
    )

    supervisor_cfg = SupervisorConfig(
        model=args.supervisor_model,
        enabled=not args.no_supervisor,
        temperature=args.temperature,
    )
    runner = BenchmarkRunner(
        providers=args.providers,
        supervisor_config=supervisor_cfg,
        max_concurrency=args.max_concurrency,
        temperature=args.temperature,
        dry_run=args.dry_run,
    )
    results = asyncio.run(runner.run(dataset))
    render_results(dataset, results)


if __name__ == "__main__":
    main()
