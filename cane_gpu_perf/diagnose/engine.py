from dataclasses import dataclass
from cane_gpu_perf.config import BenchmarkResult


@dataclass
class Finding:
    severity: str       # "critical", "warning", "info"
    category: str       # "throughput", "latency", "memory", "cost", "config", "roofline", "thermal", "efficiency", "scaling", "phase_balance"
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

        # Deep GPU diagnostics
        if result.config.deep:
            findings.extend(self._check_phase_balance(result))
            findings.extend(self._check_roofline(result))
            findings.extend(self._check_thermal(result))
            findings.extend(self._check_power_efficiency(result))
            findings.extend(self._check_multi_gpu_balance(result))
            findings.extend(self._check_memory_pressure(result))
            findings.extend(self._check_decode_performance(result))

        return sorted(findings, key=lambda f: {"critical": 0, "warning": 1, "info": 2}[f.severity])

    def diagnose_comparison(self, results: list[BenchmarkResult]) -> list[Finding]:
        """Diagnose across multiple runs - find the best config and explain why."""
        findings = []
        findings.extend(self._compare_backends(results))
        findings.extend(self._find_pareto_optimal(results))
        findings.extend(self._check_scaling_behavior(results))
        return findings

    # ----------------------------------------------------------------
    # Original diagnostics
    # ----------------------------------------------------------------

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

    # ----------------------------------------------------------------
    # Deep GPU diagnostics (require --deep / config.deep=True)
    # ----------------------------------------------------------------

    def _check_phase_balance(self, result: BenchmarkResult) -> list[Finding]:
        """Analyze prefill vs decode time distribution."""
        findings = []

        if result.prefill_fraction <= 0:
            return findings

        pf = result.prefill_fraction

        if pf > 0.5:
            findings.append(Finding(
                severity="warning",
                category="phase_balance",
                title=f"Prefill-dominated workload ({pf*100:.0f}% of time in prefill)",
                detail=f"The prefill phase (prompt processing) consumes {pf*100:.0f}% of total "
                       f"request time. This means the GPU is spending most of its time on "
                       f"compute-heavy matrix multiplications over the input, not generating tokens.",
                recommendation="Reduce prompt length, enable chunked prefill, or use prefix caching "
                              "to amortize repeated prefill work. If prompts are long, consider "
                              "FlashAttention or FlashInfer for faster attention computation.",
                expected_impact="Faster TTFT and more time spent in decode (actual generation)",
            ))
        elif pf < 0.1 and result.ttft_p50 > 0:
            findings.append(Finding(
                severity="info",
                category="phase_balance",
                title=f"Decode-dominated workload ({(1-pf)*100:.0f}% of time in decode)",
                detail=f"Prefill is only {pf*100:.0f}% of total time. "
                       f"Decode throughput ({result.decode_throughput_tps:.1f} tok/s) "
                       f"is the primary bottleneck.",
                recommendation="Optimize decode: speculative decoding, quantization (reduces "
                              "bytes per token), or multi-query/grouped-query attention.",
                expected_impact="Higher token generation throughput",
            ))

        # Report prefill and decode throughput
        if result.prefill_throughput_tps > 0 and result.decode_throughput_tps > 0:
            ratio = result.prefill_throughput_tps / result.decode_throughput_tps
            findings.append(Finding(
                severity="info",
                category="phase_balance",
                title=f"Prefill {result.prefill_throughput_tps:.0f} tok/s vs decode {result.decode_throughput_tps:.0f} tok/s ({ratio:.1f}x)",
                detail=f"Prefill throughput: {result.prefill_throughput_tps:.0f} input tok/s. "
                       f"Decode throughput: {result.decode_throughput_tps:.0f} output tok/s. "
                       f"Prefill is {ratio:.1f}x faster because it processes tokens in parallel "
                       f"(compute-bound), while decode is sequential (memory-bandwidth-bound).",
                recommendation="This ratio is expected. Focus optimization on whichever phase "
                              "dominates your workload: long prompts -> optimize prefill, "
                              "long outputs -> optimize decode.",
                expected_impact="Targeted optimization based on workload profile",
            ))

        return findings

    def _check_roofline(self, result: BenchmarkResult) -> list[Finding]:
        """Roofline model analysis - compute vs memory-bandwidth bound."""
        findings = []

        if not result.gpu_telemetry or not result.gpu_info:
            return findings

        from cane_gpu_perf.gpu.roofline import analyze_roofline

        gpu_name = result.gpu_info.get("name", "")
        prompt_tokens_mean = 0.0
        successful = [r for r in result.individual_results if r.get("error") is None]
        if successful:
            prompt_tokens_mean = sum(r.get("prompt_tokens", 0) for r in successful) / len(successful)

        # Get NVML memory utilization from telemetry
        mem_util = None
        if result.gpu_telemetry:
            snapshots = result.gpu_telemetry[0].snapshots
            if snapshots:
                mem_util = sum(s.utilization_memory for s in snapshots) / len(snapshots) / 100.0

        roofline = analyze_roofline(
            gpu_name=gpu_name,
            model_name=result.config.model,
            prefill_throughput_tps=result.prefill_throughput_tps,
            decode_throughput_tps=result.decode_throughput_tps,
            prompt_tokens_mean=prompt_tokens_mean,
            gpu_utilization_mean=result.gpu_utilization_mean,
            mem_bw_utilization=mem_util,
        )

        if roofline is None:
            return findings

        # Store roofline in result for downstream use
        result.roofline = {
            "classification": roofline.classification,
            "compute_pct": roofline.achieved_compute_pct,
            "bandwidth_pct": roofline.achieved_bandwidth_pct,
            "peak_fp16_tflops": roofline.peak_fp16_tflops,
            "peak_mem_bw_gbps": roofline.peak_mem_bw_gbps,
            "ridge_point": roofline.ridge_point,
            "bottleneck_phase": roofline.bottleneck_phase,
            "decode_bw_util": roofline.decode_bandwidth_utilization,
        }

        if roofline.classification == "memory-bound":
            findings.append(Finding(
                severity="info",
                category="roofline",
                title=f"Memory-bandwidth-bound ({roofline.achieved_bandwidth_pct:.0f}% of {roofline.peak_mem_bw_gbps:.0f} GB/s)",
                detail=f"Workload is memory-bandwidth-bound on {roofline.gpu_name}. "
                       f"Decode phase uses {roofline.decode_bandwidth_utilization*100:.0f}% of "
                       f"{roofline.peak_mem_bw_gbps:.0f} GB/s peak bandwidth. "
                       f"This is expected for autoregressive LLM generation -- each output token "
                       f"requires reading model weights from HBM.",
                recommendation="Reduce bytes per token: INT8/INT4 quantization (linear speedup for "
                              "memory-bound workloads), grouped-query attention (reduces KV cache reads), "
                              "or speculative decoding (amortizes weight reads over multiple tokens).",
                expected_impact="INT8: ~2x decode throughput. INT4: ~4x. Speculative decoding: 2-3x.",
            ))
        elif roofline.classification == "compute-bound":
            findings.append(Finding(
                severity="info",
                category="roofline",
                title=f"Compute-bound ({roofline.achieved_compute_pct:.0f}% of {roofline.peak_fp16_tflops:.0f} TFLOPS)",
                detail=f"Workload is compute-bound on {roofline.gpu_name}. "
                       f"GPU SM utilization at {roofline.achieved_compute_pct:.0f}% of peak "
                       f"{roofline.peak_fp16_tflops:.0f} FP16 TFLOPS. "
                       f"This is typical for prefill-heavy workloads with long prompts.",
                recommendation="Use FlashAttention/FlashInfer for attention computation. "
                              "Enable tensor parallelism across GPUs to distribute compute. "
                              "Consider chunked prefill to overlap prefill with decode.",
                expected_impact="FlashAttention: 2-4x prefill speedup. TP: near-linear scaling.",
            ))
        elif roofline.classification == "under-utilized":
            findings.append(Finding(
                severity="warning",
                category="roofline",
                title=f"Under-utilizing both compute ({roofline.achieved_compute_pct:.0f}%) and bandwidth ({roofline.achieved_bandwidth_pct:.0f}%)",
                detail=f"Neither compute nor memory bandwidth is near capacity on {roofline.gpu_name}. "
                       f"The GPU has {roofline.peak_fp16_tflops:.0f} TFLOPS and "
                       f"{roofline.peak_mem_bw_gbps:.0f} GB/s available but the workload isn't "
                       f"using either. This points to a CPU-side or scheduling bottleneck.",
                recommendation="Increase concurrency/batch size to feed the GPU. Check for CPU "
                              "bottlenecks (tokenization, data loading). Verify the serving framework "
                              "is using continuous batching.",
                expected_impact="2-10x throughput by saturating the GPU",
            ))

        return findings

    def _check_thermal(self, result: BenchmarkResult) -> list[Finding]:
        """Thermal throttling detection."""
        findings = []

        if not result.gpu_telemetry:
            return findings

        ts = result.gpu_telemetry[0]

        # Clock throttling
        if ts.clock_throttle_ratio > 0.1:
            findings.append(Finding(
                severity="warning" if ts.clock_throttle_ratio > 0.3 else "info",
                category="thermal",
                title=f"GPU clock throttled {ts.clock_throttle_ratio*100:.0f}% of benchmark",
                detail=f"SM clock dropped below 90% of max ({ts.clock_max_sm_mhz} MHz) during "
                       f"{ts.clock_throttle_ratio*100:.0f}% of the benchmark. "
                       f"Min clock: {ts.clock_sm_min_mhz} MHz, mean: {ts.clock_sm_mean_mhz:.0f} MHz. "
                       f"Peak temperature: {ts.temperature_peak_c}C.",
                recommendation="Check GPU cooling. If in a data center, verify airflow. "
                              "Consider lowering power limit for more stable (if slightly lower) clocks. "
                              "Sustained stable clocks often beat higher-but-throttled clocks.",
                expected_impact="More consistent latency, potentially higher sustained throughput",
            ))

        # Temperature warning
        if ts.temperature_peak_c >= 80:
            findings.append(Finding(
                severity="warning" if ts.temperature_peak_c >= 85 else "info",
                category="thermal",
                title=f"GPU peak temperature {ts.temperature_peak_c}C (mean {ts.temperature_mean_c:.0f}C)",
                detail=f"GPU reached {ts.temperature_peak_c}C during benchmark. "
                       f"Data center GPUs typically throttle at 83C. "
                       f"Mean temperature was {ts.temperature_mean_c:.0f}C.",
                recommendation="Improve cooling or reduce power limit. At these temperatures, "
                              "the GPU may be silently throttling clocks to stay within thermal envelope.",
                expected_impact="Stable performance without thermal-induced variance",
            ))

        return findings

    def _check_power_efficiency(self, result: BenchmarkResult) -> list[Finding]:
        """Power and energy efficiency analysis."""
        findings = []

        if not result.gpu_telemetry:
            return findings

        from cane_gpu_perf.gpu.efficiency import analyze_efficiency

        ts = result.gpu_telemetry[0]
        total_tokens = sum(r.get("output_tokens", 0) for r in result.individual_results
                          if r.get("error") is None)

        report = analyze_efficiency(
            gpu_telemetry=ts,
            aggregate_tps=result.aggregate_tps,
            total_tokens=total_tokens,
            duration_s=ts.duration_s,
        )

        if report is None:
            return findings

        # Store efficiency metrics in result
        result.tokens_per_watt = report.tokens_per_watt
        result.tokens_per_joule = report.tokens_per_joule
        if report.power_cost_per_1m_tokens is not None:
            result.power_cost_per_1m_tokens = report.power_cost_per_1m_tokens

        # Energy efficiency finding
        findings.append(Finding(
            severity="info",
            category="efficiency",
            title=f"Energy: {report.tokens_per_watt:.1f} tok/W, {report.tokens_per_joule:.2f} tok/J",
            detail=f"Power draw: {report.power_mean_w:.0f}W mean / {report.power_peak_w:.0f}W peak "
                   f"(limit: {report.power_limit_w:.0f}W, {report.power_headroom_pct:.0f}% headroom). "
                   f"Total energy: {report.total_energy_j:.0f}J for {report.total_tokens} tokens.",
            recommendation="Compare across quantization levels: INT8 typically doubles tok/W "
                          "for memory-bound workloads. Lower power limits can improve tok/J "
                          "at the cost of some throughput.",
            expected_impact="Quantization: ~2x energy efficiency for decode",
        ))

        # Cost estimate
        if report.power_cost_per_1m_tokens is not None:
            findings.append(Finding(
                severity="info",
                category="efficiency",
                title=f"Power cost: ${report.power_cost_per_1m_tokens:.4f}/1M tokens (at ${report.electricity_rate_kwh:.2f}/kWh)",
                detail=f"Based on {report.power_mean_w:.0f}W mean draw at "
                       f"${report.electricity_rate_kwh:.2f}/kWh, electricity alone costs "
                       f"${report.power_cost_per_1m_tokens:.4f} per 1M output tokens. "
                       f"This excludes GPU amortization, cooling, and infrastructure.",
                recommendation="For TCO analysis, multiply power cost by ~3x for full "
                              "data center overhead (cooling, networking, redundancy).",
                expected_impact="Informed capacity planning and cost modeling",
            ))

        # Power headroom
        if report.power_headroom_pct > 30:
            findings.append(Finding(
                severity="info",
                category="efficiency",
                title=f"Power headroom: {report.power_headroom_pct:.0f}% below limit",
                detail=f"GPU drawing {report.power_mean_w:.0f}W vs {report.power_limit_w:.0f}W limit. "
                       f"The GPU has significant power headroom, suggesting it could clock higher "
                       f"or handle more work.",
                recommendation="Increase concurrency to push utilization up. Alternatively, "
                              "lower the power limit to save energy without affecting throughput "
                              "(if already memory-bound).",
                expected_impact="Better energy efficiency or higher throughput",
            ))

        return findings

    def _check_multi_gpu_balance(self, result: BenchmarkResult) -> list[Finding]:
        """Multi-GPU utilization balance and topology analysis."""
        findings = []

        if not result.gpu_telemetry or len(result.gpu_telemetry) < 2:
            return findings

        from cane_gpu_perf.gpu.topology import analyze_gpu_balance

        report = analyze_gpu_balance(result.gpu_telemetry)
        if report is None:
            return findings

        # Store topology in result
        result.topology = {
            "gpu_count": report.gpu_count,
            "interconnect": report.interconnect_type,
            "gpu_names": report.gpu_names,
            "utilization_per_gpu": report.utilization_per_gpu,
            "utilization_stdev": report.utilization_stdev,
        }

        # Interconnect type
        findings.append(Finding(
            severity="info",
            category="scaling",
            title=f"{report.gpu_count} GPUs detected, {report.interconnect_type} interconnect",
            detail=f"GPUs: {', '.join(report.gpu_names)}. "
                   f"Interconnect: {report.interconnect_type}. "
                   + (f"NVLink provides ~600 GB/s bidirectional bandwidth vs ~32 GB/s for PCIe Gen4."
                      if report.interconnect_type == "nvlink" else
                      "PCIe interconnect limits tensor parallelism efficiency for large models."
                      if report.interconnect_type == "pcie" else ""),
            recommendation="NVLink is preferred for tensor parallelism. PCIe is adequate for "
                          "pipeline parallelism or independent request routing."
                          if report.interconnect_type == "pcie" else
                          "NVLink topology is optimal for tensor parallelism.",
            expected_impact="Informed parallelism strategy",
        ))

        # Utilization balance
        if report.utilization_imbalance > 0.2:
            utils_str = ", ".join(f"GPU{i}: {u*100:.0f}%"
                                  for i, u in enumerate(report.utilization_per_gpu))
            findings.append(Finding(
                severity="warning",
                category="scaling",
                title=f"GPU utilization imbalance: {report.utilization_imbalance*100:.0f}% spread",
                detail=f"Per-GPU utilization: {utils_str}. "
                       f"Standard deviation: {report.utilization_stdev*100:.1f}%. "
                       f"This suggests uneven work distribution across GPUs.",
                recommendation="Check tensor parallelism configuration. Uneven splits or "
                              "pipeline bubble overhead can cause imbalance. If using pipeline "
                              "parallelism, the last stage often has lower utilization.",
                expected_impact="Balanced utilization could improve throughput by "
                              f"{report.utilization_imbalance*100:.0f}%",
            ))

        return findings

    def _check_memory_pressure(self, result: BenchmarkResult) -> list[Finding]:
        """KV cache growth and OOM risk analysis."""
        findings = []

        if not result.gpu_telemetry:
            return findings

        ts = result.gpu_telemetry[0]

        if ts.memory_total_mb <= 0:
            return findings

        usage_pct = ts.memory_used_peak_mb / ts.memory_total_mb
        growth_mb = ts.memory_used_peak_mb - ts.memory_used_mean_mb

        if growth_mb > 0 and usage_pct > 0.8:
            findings.append(Finding(
                severity="warning" if usage_pct > 0.9 else "info",
                category="memory_pressure",
                title=f"Memory pressure: {usage_pct*100:.0f}% peak, {growth_mb:.0f}MB growth during benchmark",
                detail=f"Peak memory: {ts.memory_used_peak_mb:.0f}MB / {ts.memory_total_mb:.0f}MB "
                       f"({usage_pct*100:.0f}%). Memory grew {growth_mb:.0f}MB during the benchmark, "
                       f"likely from KV cache expansion under concurrent load.",
                recommendation="KV cache grows with concurrency * sequence_length. At current usage, "
                              "increasing concurrency may trigger OOM. Consider: PagedAttention (vLLM), "
                              "KV cache quantization, or reducing max_tokens.",
                expected_impact="Prevent OOM and maintain throughput under higher load",
            ))

        return findings

    def _check_decode_performance(self, result: BenchmarkResult) -> list[Finding]:
        """Inter-token latency analysis for decode phase."""
        findings = []

        if result.decode_latency_p50_ms <= 0:
            return findings

        itl_p50 = result.decode_latency_p50_ms
        itl_p99 = result.decode_latency_p99_ms

        # For real-time chat, inter-token latency > 100ms is noticeable
        if itl_p50 > 100:
            findings.append(Finding(
                severity="warning",
                category="phase_balance",
                title=f"Slow decode: {itl_p50:.0f}ms inter-token latency (p50)",
                detail=f"Inter-token latency: p50={itl_p50:.0f}ms, p95={result.decode_latency_p95_ms:.0f}ms, "
                       f"p99={itl_p99:.0f}ms. Users perceive token-by-token streaming as smooth below "
                       f"~50-80ms per token. At {itl_p50:.0f}ms, generation feels sluggish.",
                recommendation="For memory-bound decode: quantize (INT8/INT4), use grouped-query attention, "
                              "or try speculative decoding. For compute issues: reduce concurrent requests.",
                expected_impact="Faster perceived generation speed",
            ))

        # Decode latency variance
        if itl_p99 > itl_p50 * 4 and itl_p50 > 0:
            findings.append(Finding(
                severity="warning",
                category="phase_balance",
                title=f"Decode latency spikes (p99/p50 = {itl_p99/itl_p50:.1f}x)",
                detail=f"Inter-token latency p99 ({itl_p99:.0f}ms) is {itl_p99/itl_p50:.1f}x the "
                       f"median ({itl_p50:.0f}ms). Spikes indicate periodic stalls -- likely "
                       f"attention over growing KV cache, GC pauses, or scheduling jitter.",
                recommendation="Profile the stalls. If KV cache related, enable PagedAttention. "
                              "If GC, check Python GC settings in the serving framework.",
                expected_impact="Smoother token streaming",
            ))

        return findings

    # ----------------------------------------------------------------
    # Comparison diagnostics
    # ----------------------------------------------------------------

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
