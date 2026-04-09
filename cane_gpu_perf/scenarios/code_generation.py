"""
Simulates a coding assistant workload.
- Code completion requests with varying complexity
- Tests both short completions (autocomplete) and long generations (full functions)
- Measures quality consistency alongside performance
"""

from cane_gpu_perf.scenarios.base import Scenario
from cane_gpu_perf.config import BenchmarkConfig


def code_generation_scenario(model: str, backend: str) -> Scenario:
    return Scenario(
        name="Code Generation",
        description=(
            "Simulates an AI coding assistant handling autocomplete (short, fast) "
            "and full function generation (longer, complex). Tests whether the model "
            "maintains quality under different output length requirements."
        ),
        latency_budget_ms=2000,
        success_criteria="Autocomplete TTFT < 300ms. Full generation < 2000ms p95.",
        configs=[
            # Autocomplete (short, fast)
            BenchmarkConfig(
                model=model, backend=backend,
                num_requests=100, concurrency=8,
                max_tokens=50, prompt_set="code",
            ),
            # Full function generation
            BenchmarkConfig(
                model=model, backend=backend,
                num_requests=50, concurrency=4,
                max_tokens=512, prompt_set="code",
            ),
        ],
    )
