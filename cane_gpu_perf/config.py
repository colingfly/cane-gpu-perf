"""Core configuration and result dataclasses."""

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

    # GPU metrics
    gpu_utilization_mean: float | None = None
    peak_gpu_memory_gb: float | None = None
    gpu_info: dict | None = None

    # Cost
    cost_per_1k_tokens: float | None = None

    # Raw data
    individual_results: list = field(default_factory=list)
