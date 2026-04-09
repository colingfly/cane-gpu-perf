from dataclasses import dataclass
from cane_gpu_perf.config import BenchmarkConfig


@dataclass
class Scenario:
    name: str
    description: str
    configs: list[BenchmarkConfig]
    latency_budget_ms: float | None = None  # target latency SLA
    throughput_target: float | None = None   # target tok/s
    success_criteria: str = ""
