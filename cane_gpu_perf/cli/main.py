"""CLI entry point for cane-perf."""

import argparse
import asyncio


def _print_gpu_telemetry(console, result):
    """Print GPU telemetry summary if available."""
    if not result.gpu_telemetry:
        return

    from rich.table import Table

    for ts in result.gpu_telemetry:
        table = Table(title=f"GPU {ts.gpu_index}: {ts.gpu_name}", show_header=True, border_style="dim")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Detail", style="dim")

        table.add_row(
            "Utilization",
            f"{ts.utilization_mean*100:.0f}%",
            f"min={ts.utilization_min*100:.0f}% p95={ts.utilization_p95*100:.0f}%",
        )
        table.add_row(
            "Memory",
            f"{ts.memory_used_peak_mb/1024:.1f} / {ts.memory_total_mb/1024:.1f} GB",
            f"peak={ts.memory_used_peak_mb/1024:.1f}GB mean={ts.memory_used_mean_mb/1024:.1f}GB",
        )
        table.add_row(
            "Temperature",
            f"{ts.temperature_peak_c}C peak",
            f"mean={ts.temperature_mean_c:.0f}C",
        )
        table.add_row(
            "Power",
            f"{ts.power_draw_mean_w:.0f}W mean",
            f"peak={ts.power_draw_peak_w:.0f}W limit={ts.power_limit_w:.0f}W",
        )
        table.add_row(
            "SM Clock",
            f"{ts.clock_sm_mean_mhz:.0f} MHz mean",
            f"min={ts.clock_sm_min_mhz} max={ts.clock_max_sm_mhz} MHz"
            + (f" throttled={ts.clock_throttle_ratio*100:.0f}%" if ts.clock_throttle_ratio > 0 else ""),
        )
        table.add_row(
            "PCIe",
            f"TX={ts.pcie_tx_peak_kbps/1024:.0f} MB/s peak",
            f"RX={ts.pcie_rx_peak_kbps/1024:.0f} MB/s peak",
        )
        table.add_row(
            "Samples",
            f"{ts.num_samples}",
            f"over {ts.duration_s:.1f}s",
        )

        console.print(table)


def _print_phase_metrics(console, result):
    """Print prefill/decode phase separation metrics."""
    if result.prefill_throughput_tps <= 0 and result.decode_throughput_tps <= 0:
        return

    console.print(f"\n[bold]Phase Analysis:[/bold]")
    if result.prefill_throughput_tps > 0:
        console.print(f"  Prefill:   {result.prefill_throughput_tps:.0f} tok/s  "
                      f"({result.prefill_fraction*100:.0f}% of request time)")
    if result.decode_throughput_tps > 0:
        console.print(f"  Decode:    {result.decode_throughput_tps:.0f} tok/s  "
                      f"ITL p50={result.decode_latency_p50_ms:.0f}ms "
                      f"p95={result.decode_latency_p95_ms:.0f}ms "
                      f"p99={result.decode_latency_p99_ms:.0f}ms")


def _print_efficiency(console, result):
    """Print power/efficiency metrics."""
    if result.tokens_per_watt is None:
        return

    console.print(f"\n[bold]Efficiency:[/bold]")
    console.print(f"  Energy:    {result.tokens_per_watt:.1f} tok/W  "
                  f"{result.tokens_per_joule:.2f} tok/J" if result.tokens_per_joule else "")
    if result.power_cost_per_1m_tokens is not None:
        console.print(f"  Power cost: ${result.power_cost_per_1m_tokens:.4f}/1M tokens")


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
        deep=args.deep,
    )

    runner = BenchmarkRunner()
    result = asyncio.run(runner.run(config))

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Requests:  {result.total_requests} total, {result.failed_requests} failed")
    console.print(f"  Latency:   p50={result.latency_p50:.0f}ms  p95={result.latency_p95:.0f}ms  p99={result.latency_p99:.0f}ms")
    console.print(f"  TTFT:      p50={result.ttft_p50:.0f}ms  p95={result.ttft_p95:.0f}ms  p99={result.ttft_p99:.0f}ms")
    console.print(f"  Throughput: {result.aggregate_tps:.1f} tok/s aggregate, {result.tokens_per_second_mean:.1f} tok/s mean per-request")
    console.print(f"  RPS:       {result.requests_per_second:.2f} req/s")

    if result.failed_requests > 0:
        errors = [r["error"] for r in result.individual_results if r.get("error")]
        unique_errors = list(set(errors))
        console.print(f"\n[bold]Errors ({len(errors)} total, {len(unique_errors)} unique):[/bold]")
        for err in unique_errors[:5]:
            console.print(f"  - {err}")

    # Deep mode: phase analysis, GPU telemetry, efficiency
    if args.deep:
        _print_phase_metrics(console, result)
        _print_gpu_telemetry(console, result)
        _print_efficiency(console, result)

    if args.diagnose or args.deep:
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

    # If --deep, enable deep mode on all scenario configs
    if args.deep:
        for config in scenario.configs:
            config.deep = True

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
    bench_parser.add_argument("--deep", action="store_true",
                             help="Enable deep GPU analysis: NVML telemetry, roofline model, "
                                  "power/thermal efficiency, prefill/decode separation")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run workload scenario with diagnosis")
    analyze_parser.add_argument("--model", required=True, help="Model name")
    analyze_parser.add_argument("--backend", required=True,
                               choices=["vllm", "sglang", "ollama", "openrouter", "modal", "openai"])
    analyze_parser.add_argument("--scenario", required=True,
                               choices=["chatbot", "rag", "batch", "code"],
                               help="Workload scenario to simulate")
    analyze_parser.add_argument("--output", help="Save results to JSON")
    analyze_parser.add_argument("--deep", action="store_true",
                               help="Enable deep GPU analysis: NVML telemetry, roofline model, "
                                    "power/thermal efficiency, prefill/decode separation")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    elif args.command == "bench":
        cmd_bench(args)
    elif args.command == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()
