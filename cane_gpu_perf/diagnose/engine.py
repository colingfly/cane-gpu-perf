from dataclasses import dataclass
from cane_gpu_perf.config import BenchmarkResult


@dataclass
class Finding:
    severity: str       # "critical", "warning", "info"
    category: str       # "throughput", "latency", "memory", "cost", "config"
    title: str          # one-line summary
    detail: str         # explanation
    recommendation: str # what to do about it
    expected_impact: str # "2-3x throughput improvement"


class DiagnoseEngine:
    """
    Reads benchmark results and produces opinionated findings.
    This is the FDE layer - not just measuring, but diagnosing.
    """

    def diagnose(self, result: BenchmarkResult) -> list[Finding]:
        findings = []
        findings.extend(self._check_gpu_utilization(result))
        findings.extend(self._check_batching(result))
        findings.extend(self._check_ttft(result))
        findings.extend(self._check_latency_variance(result))
        findings.extend(self._check_failure_rate(result))
        findings.extend(self._check_throughput_ceiling(result))
        findings.extend(self._check_memory(result))
        findings.extend(self._check_cost_efficiency(result))
        findings.extend(self._check_context_scaling(result))
        return sorted(findings, key=lambda f: {"critical": 0, "warning": 1, "info": 2}[f.severity])

    def diagnose_comparison(self, results: list[BenchmarkResult]) -> list[Finding]:
        """Diagnose across multiple runs - find the best config and explain why."""
        findings = []
        findings.extend(self._compare_backends(results))
        findings.extend(self._find_pareto_optimal(results))
        findings.extend(self._check_scaling_behavior(results))
        return findings

    def _check_gpu_utilization(self, result: BenchmarkResult) -> list[Finding]:
        findings = []
        util = result.gpu_utilization_mean
        if util is None:
            return findings

        if util < 0.4:
            findings.append(Finding(
                severity="critical",
                category="throughput",
                title=f"GPU severely under-utilized ({util*100:.0f}%)",
                detail=f"GPU utilization averaged {util*100:.0f}% during the benchmark. "
                       f"This means the GPU is idle more than half the time, likely waiting "
                       f"for data or bottlenecked by CPU-side processing.",
                recommendation="Increase batch size or concurrency to keep the GPU fed. "
                              "Try doubling concurrency first. If using vLLM, enable "
                              "continuous batching with --enable-chunked-prefill.",
                expected_impact="2-4x throughput improvement",
            ))
        elif util < 0.7:
            findings.append(Finding(
                severity="warning",
                category="throughput",
                title=f"GPU under-utilized ({util*100:.0f}%)",
                detail=f"GPU utilization at {util*100:.0f}%. There's room to push more work "
                       f"through without hitting GPU limits.",
                recommendation="Increase concurrency or batch size. Target 80-90% utilization.",
                expected_impact="1.3-2x throughput improvement",
            ))
        elif util > 0.95:
            findings.append(Finding(
                severity="info",
                category="throughput",
                title=f"GPU fully saturated ({util*100:.0f}%)",
                detail=f"GPU is running at near-maximum utilization. Further throughput gains "
                       f"require a faster GPU, model optimization, or quantization.",
                recommendation="Consider INT8 quantization if quality allows, or scale to "
                              "multiple GPUs with tensor parallelism.",
                expected_impact="Quantization: 1.5-2x. Multi-GPU: near-linear scaling.",
            ))
        return findings

    def _check_batching(self, result: BenchmarkResult) -> list[Finding]:
        findings = []
        cfg = result.config

        if cfg.concurrency == 1 and result.requests_per_second < 2.0:
            findings.append(Finding(
                severity="warning",
                category="config",
                title="Running without concurrency",
                detail=f"Benchmark ran with concurrency=1, achieving {result.requests_per_second:.1f} req/s. "
                       f"Sequential requests can't saturate the GPU.",
                recommendation="Re-run with --concurrency 4, 8, or 16 to find the throughput ceiling. "
                              "Most serving backends handle concurrent requests efficiently.",
                expected_impact="2-8x throughput depending on model and hardware",
            ))

        if cfg.concurrency > 1:
            theoretical_max = result.tokens_per_second_mean * cfg.concurrency
            actual = result.aggregate_tps
            efficiency = actual / theoretical_max if theoretical_max > 0 else 0

            if efficiency < 0.5:
                findings.append(Finding(
                    severity="warning",
                    category="throughput",
                    title=f"Poor concurrency scaling ({efficiency*100:.0f}% efficient)",
                    detail=f"With {cfg.concurrency} concurrent requests, aggregate throughput "
                           f"is only {efficiency*100:.0f}% of theoretical max. Requests are "
                           f"likely queuing or contending for resources.",
                    recommendation="Try reducing concurrency or increasing GPU memory allocation. "
                                  "Check if KV cache is being evicted under load.",
                    expected_impact="Better resource utilization at current concurrency",
                ))
        return findings

    def _check_ttft(self, result: BenchmarkResult) -> list[Finding]:
        findings = []

        if result.ttft_p50 > 2000:
            findings.append(Finding(
                severity="critical",
                category="latency",
                title=f"Very high TTFT ({result.ttft_p50:.0f}ms p50)",
                detail=f"Time to first token is {result.ttft_p50:.0f}ms at p50, "
                       f"{result.ttft_p95:.0f}ms at p95. Users perceive anything over "
                       f"1-2 seconds as broken. This is likely model loading, cold start, "
                       f"or prefill bottleneck.",
                recommendation="If cold start: use persistent serving (keep model loaded). "
                              "If prefill: reduce prompt length or enable chunked prefill. "
                              "If using Modal: check container warm-up strategy.",
                expected_impact="10-50x TTFT reduction with warm model",
            ))
        elif result.ttft_p50 > 500:
            findings.append(Finding(
                severity="warning",
                category="latency",
                title=f"Elevated TTFT ({result.ttft_p50:.0f}ms p50)",
                detail=f"TTFT at {result.ttft_p50:.0f}ms is acceptable but not great. "
                       f"For chat applications, target <300ms.",
                recommendation="Check if prompt is being re-tokenized per request. "
                              "Enable prefix caching if the serving backend supports it.",
                expected_impact="30-50% TTFT reduction",
            ))

        if result.ttft_p50 > 0 and result.ttft_p99 > result.ttft_p50 * 5:
            findings.append(Finding(
                severity="warning",
                category="latency",
                title=f"High TTFT variance (p99/p50 = {result.ttft_p99/result.ttft_p50:.1f}x)",
                detail=f"TTFT ranges from {result.ttft_p50:.0f}ms (p50) to "
                       f"{result.ttft_p99:.0f}ms (p99). This suggests inconsistent behavior "
                       f"- possibly cold starts, GC pauses, or request queuing.",
                recommendation="Investigate the slow outliers. If using serverless (Modal), "
                              "consider keep-alive to avoid cold starts on the tail.",
                expected_impact="More predictable user experience",
            ))
        return findings

    def _check_latency_variance(self, result: BenchmarkResult) -> list[Finding]:
        findings = []
        ratio = result.latency_p99 / result.latency_p50 if result.latency_p50 > 0 else 0

        if ratio > 5:
            findings.append(Finding(
                severity="warning",
                category="latency",
                title=f"High latency variance (p99/p50 = {ratio:.1f}x)",
                detail=f"Latency p50={result.latency_p50:.0f}ms but p99={result.latency_p99:.0f}ms. "
                       f"The tail is {ratio:.1f}x worse than median. Some requests are hitting "
                       f"a fundamentally different code path.",
                recommendation="Profile the slow requests. Common causes: KV cache eviction, "
                              "request queuing, variable output length, or GC pauses.",
                expected_impact="Tighter SLA compliance",
            ))
        return findings

    def _check_failure_rate(self, result: BenchmarkResult) -> list[Finding]:
        findings = []
        fail_rate = result.failed_requests / result.total_requests if result.total_requests > 0 else 0

        if fail_rate > 0.1:
            findings.append(Finding(
                severity="critical",
                category="reliability",
                title=f"High failure rate ({fail_rate*100:.1f}%)",
                detail=f"{result.failed_requests} out of {result.total_requests} requests failed. "
                       f"This is unacceptable for production.",
                recommendation="Check error messages in individual results. Common causes: "
                              "OOM, timeout, rate limiting, or backend instability.",
                expected_impact="Eliminating failures is prerequisite for production",
            ))
        elif fail_rate > 0.01:
            findings.append(Finding(
                severity="warning",
                category="reliability",
                title=f"Some failures ({fail_rate*100:.1f}%)",
                detail=f"{result.failed_requests} requests failed out of {result.total_requests}.",
                recommendation="Investigate the error patterns. Consider retry logic.",
                expected_impact="Improved reliability",
            ))
        return findings

    def _check_throughput_ceiling(self, result: BenchmarkResult) -> list[Finding]:
        findings = []

        if result.aggregate_tps < 10 and result.config.concurrency >= 4:
            findings.append(Finding(
                severity="warning",
                category="throughput",
                title=f"Low aggregate throughput ({result.aggregate_tps:.1f} tok/s)",
                detail=f"Even with concurrency={result.config.concurrency}, aggregate throughput "
                       f"is only {result.aggregate_tps:.1f} tokens/sec. This suggests a "
                       f"fundamental bottleneck.",
                recommendation="Check if the model fits in GPU memory without offloading. "
                              "Consider a smaller model, quantization, or faster hardware.",
                expected_impact="Depends on root cause",
            ))
        return findings

    def _check_memory(self, result: BenchmarkResult) -> list[Finding]:
        findings = []
        if result.peak_gpu_memory_gb and result.gpu_info:
            total = result.gpu_info.get("memory_total_gb", 0)
            used = result.peak_gpu_memory_gb
            if total > 0:
                usage_pct = used / total
                if usage_pct > 0.95:
                    findings.append(Finding(
                        severity="critical",
                        category="memory",
                        title=f"GPU memory nearly full ({usage_pct*100:.0f}%)",
                        detail=f"Peak memory usage: {used:.1f}GB out of {total:.1f}GB. "
                               f"You're at risk of OOM under load. No room for KV cache growth.",
                        recommendation="Reduce batch size, enable quantization (INT8 saves ~50% memory), "
                                      "or move to a GPU with more VRAM.",
                        expected_impact="INT8: ~50% memory reduction. INT4: ~75%.",
                    ))
                elif usage_pct < 0.5:
                    findings.append(Finding(
                        severity="info",
                        category="memory",
                        title=f"GPU memory under-utilized ({usage_pct*100:.0f}%)",
                        detail=f"Only using {used:.1f}GB of {total:.1f}GB. You have headroom "
                               f"to increase batch size or serve longer contexts.",
                        recommendation="Increase batch size to use available memory for throughput.",
                        expected_impact="Better hardware utilization",
                    ))
        return findings

    def _check_cost_efficiency(self, result: BenchmarkResult) -> list[Finding]:
        findings = []
        if result.cost_per_1k_tokens:
            if result.cost_per_1k_tokens > 0.01:
                findings.append(Finding(
                    severity="info",
                    category="cost",
                    title=f"Cost: ${result.cost_per_1k_tokens:.4f}/1K tokens",
                    detail=f"At current throughput and pricing, inference costs "
                           f"${result.cost_per_1k_tokens:.4f} per 1K output tokens.",
                    recommendation="Compare with self-hosted options. At this price point, "
                                  f"self-hosting on a single A100 typically breaks even at "
                                  f"~{int(50 / (result.cost_per_1k_tokens * 1000))}K tokens/day.",
                    expected_impact="Potential 2-10x cost reduction with self-hosting at scale",
                ))
        return findings

    def _check_context_scaling(self, result: BenchmarkResult) -> list[Finding]:
        return []

    def _compare_backends(self, results: list[BenchmarkResult]) -> list[Finding]:
        """Compare multiple backend results and declare winners."""
        findings = []
        if len(results) < 2:
            return findings

        by_latency = sorted(results, key=lambda r: r.latency_p50)
        by_throughput = sorted(results, key=lambda r: r.aggregate_tps, reverse=True)

        best_latency = by_latency[0]
        best_throughput = by_throughput[0]

        findings.append(Finding(
            severity="info",
            category="comparison",
            title=f"Lowest latency: {best_latency.config.backend} ({best_latency.latency_p50:.0f}ms p50)",
            detail=f"{best_latency.config.backend} wins on latency at {best_latency.latency_p50:.0f}ms p50, "
                   f"vs {by_latency[-1].config.backend} at {by_latency[-1].latency_p50:.0f}ms. "
                   f"That's a {by_latency[-1].latency_p50/best_latency.latency_p50:.1f}x difference.",
            recommendation=f"Use {best_latency.config.backend} for latency-sensitive workloads (chat, real-time).",
            expected_impact="Best user-perceived responsiveness",
        ))

        findings.append(Finding(
            severity="info",
            category="comparison",
            title=f"Highest throughput: {best_throughput.config.backend} ({best_throughput.aggregate_tps:.0f} tok/s)",
            detail=f"{best_throughput.config.backend} wins on throughput at {best_throughput.aggregate_tps:.0f} tok/s, "
                   f"vs {by_throughput[-1].config.backend} at {by_throughput[-1].aggregate_tps:.0f} tok/s.",
            recommendation=f"Use {best_throughput.config.backend} for batch workloads (offline processing, bulk inference).",
            expected_impact="Maximum tokens per dollar",
        ))

        if best_latency.config.backend != best_throughput.config.backend:
            findings.append(Finding(
                severity="warning",
                category="comparison",
                title="Latency vs throughput tradeoff detected",
                detail=f"{best_latency.config.backend} is fastest per-request but "
                       f"{best_throughput.config.backend} moves more total tokens. "
                       f"This is the classic latency-throughput tradeoff.",
                recommendation="Choose based on workload: real-time chat -> optimize latency. "
                              "Batch processing -> optimize throughput. "
                              "Mixed -> use both backends with routing.",
                expected_impact="Right backend for each workload type",
            ))
        return findings

    def _find_pareto_optimal(self, results: list[BenchmarkResult]) -> list[Finding]:
        """Identify Pareto-optimal configurations across latency, throughput, cost."""
        findings = []
        pareto = []

        for i, r in enumerate(results):
            dominated = False
            for j, other in enumerate(results):
                if i == j:
                    continue
                if (other.latency_p50 <= r.latency_p50 and
                    other.aggregate_tps >= r.aggregate_tps and
                    (other.latency_p50 < r.latency_p50 or other.aggregate_tps > r.aggregate_tps)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)

        if pareto:
            names = [f"{r.config.backend}" for r in pareto]
            findings.append(Finding(
                severity="info",
                category="comparison",
                title=f"Pareto-optimal configs: {', '.join(names)}",
                detail=f"Out of {len(results)} configurations tested, {len(pareto)} are on the "
                       f"Pareto frontier (no other config is better on ALL dimensions). "
                       f"Configs NOT on the frontier are strictly dominated and should not be used.",
                recommendation=f"Choose from: {', '.join(names)} based on your priority "
                              f"(latency vs throughput vs cost).",
                expected_impact="Eliminate suboptimal configurations",
            ))
        return findings

    def _check_scaling_behavior(self, results: list[BenchmarkResult]) -> list[Finding]:
        """If results have different concurrency levels, analyze scaling."""
        findings = []
        by_concurrency: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            if r.config.backend not in by_concurrency:
                by_concurrency[r.config.backend] = []
            by_concurrency[r.config.backend].append(r)

        for backend, runs in by_concurrency.items():
            if len(runs) < 2:
                continue
            runs.sort(key=lambda r: r.config.concurrency)
            low = runs[0]
            high = runs[-1]

            if low.config.concurrency != high.config.concurrency:
                scaling = high.aggregate_tps / low.aggregate_tps if low.aggregate_tps > 0 else 0
                concurrency_ratio = high.config.concurrency / low.config.concurrency

                findings.append(Finding(
                    severity="info",
                    category="scaling",
                    title=f"{backend}: {scaling:.1f}x throughput at {concurrency_ratio:.0f}x concurrency",
                    detail=f"Going from concurrency={low.config.concurrency} to "
                           f"{high.config.concurrency}, throughput went from "
                           f"{low.aggregate_tps:.0f} to {high.aggregate_tps:.0f} tok/s "
                           f"({scaling:.1f}x). Linear scaling would be {concurrency_ratio:.0f}x.",
                    recommendation="Sub-linear scaling suggests a bottleneck (memory, compute, or queuing). "
                                  "Super-linear scaling suggests batching efficiency gains." if scaling < concurrency_ratio
                                  else "Good scaling efficiency. Consider pushing concurrency higher.",
                    expected_impact=f"Scaling efficiency: {scaling/concurrency_ratio*100:.0f}%",
                ))
        return findings


def format_findings(findings: list[Finding]) -> str:
    """Format findings as a readable report."""
    if not findings:
        return "No significant findings. Performance looks healthy."

    icons = {"critical": "CRITICAL:", "warning": "WARNING:", "info": "INFO:"}
    lines = ["\n## Findings\n"]

    for i, f in enumerate(findings, 1):
        lines.append(f"{icons[f.severity]} **{f.title}**")
        lines.append(f"   {f.detail}")
        lines.append(f"   -> {f.recommendation}")
        lines.append(f"   Expected impact: {f.expected_impact}")
        lines.append("")

    return "\n".join(lines)
