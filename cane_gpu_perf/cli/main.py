"""CLI entry point for cane-perf."""

import argparse
import asyncio


def cmd_bench(args):
    """Run a single benchmark."""
    from cane_gpu_perf.config import BenchmarkConfig
    from cane_gpu_perf.bench.runner import BenchmarkRunner
    from cane_gpu_perf.diagnose.engine import DiagnoseEngine, format_findings
    from rich.console import Console

    console = Console()

    config = BenchmarkConfig(
        model=args.model,
        backend=args.backend,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        prompt_set=args.prompt_set,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    runner = BenchmarkRunner()
    result = asyncio.run(runner.run(config))

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Requests:  {result.total_requests} total, {result.failed_requests} failed")
    console.print(f"  Latency:   p50={result.latency_p50:.0f}ms  p95={result.latency_p95:.0f}ms  p99={result.latency_p99:.0f}ms")
    console.print(f"  TTFT:      p50={result.ttft_p50:.0f}ms  p95={result.ttft_p95:.0f}ms  p99={result.ttft_p99:.0f}ms")
    console.print(f"  Throughput: {result.aggregate_tps:.1f} tok/s aggregate, {result.tokens_per_second_mean:.1f} tok/s mean per-request")
    console.print(f"  RPS:       {result.requests_per_second:.2f} req/s")

    if args.diagnose:
        engine = DiagnoseEngine()
        findings = engine.diagnose(result)
        console.print(format_findings(findings))


def cmd_analyze(args):
    """Run a workload scenario and produce findings."""
    from cane_gpu_perf.scenarios.runner import ScenarioRunner
    from cane_gpu_perf.scenarios.chatbot import chatbot_scenario
    from cane_gpu_perf.scenarios.rag_pipeline import rag_pipeline_scenario
    from cane_gpu_perf.scenarios.batch_processing import batch_processing_scenario
    from cane_gpu_perf.scenarios.code_generation import code_generation_scenario

    SCENARIOS = {
        "chatbot": chatbot_scenario,
        "rag": rag_pipeline_scenario,
        "batch": batch_processing_scenario,
        "code": code_generation_scenario,
    }

    if args.scenario not in SCENARIOS:
        print(f"Unknown scenario: {args.scenario}")
        print(f"Available: {', '.join(SCENARIOS.keys())}")
        return

    scenario = SCENARIOS[args.scenario](model=args.model, backend=args.backend)
    runner = ScenarioRunner()
    result = asyncio.run(runner.run(scenario))

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump({
                "scenario": result["scenario"],
                "passed": result["passed"],
                "findings": [
                    {"severity": fi.severity, "title": fi.title,
                     "detail": fi.detail, "recommendation": fi.recommendation}
                    for fi in result["findings"]
                ],
            }, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog="cane-perf",
        description="GPU inference benchmarking with opinionated diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # bench command
    bench_parser = subparsers.add_parser("bench", help="Run a single benchmark")
    bench_parser.add_argument("--model", required=True, help="Model name")
    bench_parser.add_argument("--backend", required=True,
                             choices=["vllm", "sglang", "ollama", "openrouter", "modal", "openai"])
    bench_parser.add_argument("--num-requests", type=int, default=100, help="Number of requests")
    bench_parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests")
    bench_parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens")
    bench_parser.add_argument("--prompt-set", default="default",
                             choices=["default", "conversation", "rag", "short", "code"],
                             help="Prompt set to use")
    bench_parser.add_argument("--base-url", help="Custom API base URL")
    bench_parser.add_argument("--api-key", help="API key (or set via env var)")
    bench_parser.add_argument("--diagnose", action="store_true", help="Run diagnostic analysis on results")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run workload scenario with diagnosis")
    analyze_parser.add_argument("--model", required=True, help="Model name")
    analyze_parser.add_argument("--backend", required=True,
                               choices=["vllm", "sglang", "ollama", "openrouter", "modal", "openai"])
    analyze_parser.add_argument("--scenario", required=True,
                               choices=["chatbot", "rag", "batch", "code"],
                               help="Workload scenario to simulate")
    analyze_parser.add_argument("--output", help="Save results to JSON")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    elif args.command == "bench":
        cmd_bench(args)
    elif args.command == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()
