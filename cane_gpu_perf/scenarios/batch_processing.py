"""
Simulates offline batch processing.
- 10K documents to process
- Maximize throughput (latency doesn't matter)
- Measure total job completion time and cost
"""

from cane_gpu_perf.scenarios.base import Scenario
from cane_gpu_perf.config import BenchmarkConfig


def batch_processing_scenario(model: str, backend: str) -> Scenario:
    return Scenario(
        name="Batch Processing",
        description=(
            "Simulates offline document processing where latency doesn't matter "
            "but throughput and cost do. Processes 10K items with maximum concurrency. "
            "Goal: find the concurrency level that maximizes aggregate tok/s."
        ),
        throughput_target=1000,  # tok/s target
        success_criteria="Maximize aggregate tok/s. Minimize total cost.",
        configs=[
            # Concurrency sweep to find optimal
            BenchmarkConfig(
                model=model, backend=backend,
                num_requests=200, concurrency=1,
                max_tokens=128, prompt_set="short",
            ),
            BenchmarkConfig(
                model=model, backend=backend,
                num_requests=200, concurrency=8,
                max_tokens=128, prompt_set="short",
            ),
            BenchmarkConfig(
                model=model, backend=backend,
                num_requests=200, concurrency=32,
                max_tokens=128, prompt_set="short",
            ),
            BenchmarkConfig(
                model=model, backend=backend,
                num_requests=200, concurrency=64,
                max_tokens=128, prompt_set="short",
            ),
        ],
    )
