"""Core configuration and result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    model: str
    backend: str
    num_requests: int = 100
    concurrency: int = 1
    max_tokens: int = 256
    prompt_set: str = "default"
    context_length: int | None = None
    temperature: float = 0.0
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 120.0
    deep: bool = False  # enable GPU telemetry + hardware diagnostics


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig

    # Latency metrics (ms)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    # Time to first token (ms)
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0

    # Throughput
    tokens_per_second_mean: float = 0.0
    aggregate_tps: float = 0.0
    requests_per_second: float = 0.0

    # Request counts
    total_requests: int = 0
    failed_requests: int = 0

    # -- Prefill / decode phase separation --
    prefill_throughput_tps: float = 0.0      # input tokens / TTFT (aggregate)
    decode_throughput_tps: float = 0.0       # output tokens / decode time (aggregate)
    decode_latency_p50_ms: float = 0.0       # inter-token latency
    decode_latency_p95_ms: float = 0.0
    decode_latency_p99_ms: float = 0.0
    prefill_fraction: float = 0.0            # TTFT / total latency (mean)

    # -- GPU telemetry (populated when deep=True) --
    gpu_telemetry: list | None = None  # list[GpuTimeSeries]
    gpu_utilization_mean: float | None = None
    peak_gpu_memory_gb: float | None = None
    gpu_info: dict | None = None

    # -- Roofline analysis --
    roofline: dict | None = None  # {"classification", "compute_pct", "bandwidth_pct", ...}

    # -- Power & efficiency --
    tokens_per_watt: float | None = None
    tokens_per_joule: float | None = None
    power_cost_per_1m_tokens: float | None = None
    thermal_headroom_c: float | None = None
    clock_throttle_ratio: float | None = None

    # -- Multi-GPU topology --
    topology: dict | None = None  # {"gpu_count", "interconnect", "balance", ...}

    # Cost
    cost_per_1k_tokens: float | None = None

    # Raw data
    individual_results: list = field(default_factory=list)
