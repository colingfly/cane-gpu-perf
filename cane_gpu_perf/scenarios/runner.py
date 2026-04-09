import asyncio
from rich.console import Console
from rich.panel import Panel

from cane_gpu_perf.scenarios.base import Scenario
from cane_gpu_perf.bench.runner import BenchmarkRunner
from cane_gpu_perf.diagnose.engine import DiagnoseEngine, Finding, format_findings
from cane_gpu_perf.config import BenchmarkResult

console = Console()


class ScenarioRunner:
    """Runs a complete workload scenario and produces findings."""

    def __init__(self):
        self.runner = BenchmarkRunner()
        self.diagnoser = DiagnoseEngine()

    async def run(self, scenario: Scenario) -> dict:
        console.print(Panel(
            f"[bold]{scenario.name}[/bold]\n\n{scenario.description}\n\n"
            f"Success criteria: {scenario.success_criteria}",
            title="Scenario",
            border_style="cyan",
        ))

        results: list[BenchmarkResult] = []

        for i, config in enumerate(scenario.configs):
            console.print(f"\n[bold]Phase {i+1}/{len(scenario.configs)}[/bold]")
            result = await self.runner.run(config)
            results.append(result)

        # Diagnose individual results
        all_findings = []
        for result in results:
            findings = self.diagnoser.diagnose(result)
            all_findings.extend(findings)

        # Diagnose across results
        if len(results) > 1:
            comparison_findings = self.diagnoser.diagnose_comparison(results)
            all_findings.extend(comparison_findings)

        # Check SLA compliance
        sla_findings = self._check_sla(scenario, results)
        all_findings.extend(sla_findings)

        # Print findings
        console.print(format_findings(all_findings))

        # Summary verdict
        passed = all(f.severity != "critical" for f in sla_findings)
        if passed:
            console.print(Panel("[bold green]\u2705 PASS[/bold green] \u2014 All success criteria met.",
                               border_style="green"))
        else:
            console.print(Panel("[bold red]\u274c FAIL[/bold red] \u2014 One or more criteria not met.",
                               border_style="red"))

        return {
            "scenario": scenario.name,
            "results": results,
            "findings": all_findings,
            "passed": passed,
        }

    def _check_sla(self, scenario: Scenario, results: list[BenchmarkResult]) -> list[Finding]:
        findings = []

        if scenario.latency_budget_ms:
            worst_ttft_p95 = max(r.ttft_p95 for r in results)
            if worst_ttft_p95 > scenario.latency_budget_ms:
                findings.append(Finding(
                    severity="critical",
                    category="sla",
                    title=f"SLA BREACH: TTFT p95 ({worst_ttft_p95:.0f}ms) exceeds budget ({scenario.latency_budget_ms:.0f}ms)",
                    detail=f"The latency SLA requires TTFT p95 < {scenario.latency_budget_ms:.0f}ms "
                           f"but the worst phase hit {worst_ttft_p95:.0f}ms.",
                    recommendation="Reduce concurrency, optimize prefill, or use a smaller/faster model.",
                    expected_impact="Meet latency SLA",
                ))
            else:
                findings.append(Finding(
                    severity="info",
                    category="sla",
                    title=f"SLA MET: TTFT p95 ({worst_ttft_p95:.0f}ms) within budget ({scenario.latency_budget_ms:.0f}ms)",
                    detail="Latency SLA is being met across all load phases.",
                    recommendation="Consider increasing concurrency to improve throughput while staying within SLA.",
                    expected_impact="More throughput at same latency",
                ))

        if scenario.throughput_target:
            best_tps = max(r.aggregate_tps for r in results)
            if best_tps < scenario.throughput_target:
                findings.append(Finding(
                    severity="critical",
                    category="sla",
                    title=f"Throughput target missed: {best_tps:.0f} tok/s vs {scenario.throughput_target:.0f} target",
                    detail=f"Best aggregate throughput was {best_tps:.0f} tok/s, "
                           f"below the target of {scenario.throughput_target:.0f}.",
                    recommendation="Increase concurrency, enable batching, or use a faster backend.",
                    expected_impact="Meet throughput requirements",
                ))

        return findings
