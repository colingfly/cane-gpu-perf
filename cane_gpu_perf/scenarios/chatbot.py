"""
Simulates a real-time chat application.
- 50 concurrent users
- 5 messages per session (multi-turn)
- Latency budget: p95 < 1500ms TTFT
- Mix of short and medium prompts
"""

from cane_gpu_perf.scenarios.base import Scenario
from cane_gpu_perf.config import BenchmarkConfig


def chatbot_scenario(model: str, backend: str) -> Scenario:
    return Scenario(
        name="Real-Time Chatbot",
        description=(
            "Simulates a production chat application with 50 concurrent users. "
            "Each user sends 5 messages in a session. The system must maintain "
            "sub-1.5s TTFT at p95 while handling concurrent load."
        ),
        latency_budget_ms=1500,
        success_criteria="TTFT p95 < 1500ms AND failure rate < 1%",
        configs=[
            # Phase 1: Light load (10 concurrent)
            BenchmarkConfig(
                model=model, backend=backend,
                num_requests=50, concurrency=10,
                max_tokens=256, prompt_set="conversation",
            ),
            # Phase 2: Medium load (25 concurrent)
            BenchmarkConfig(
                model=model, backend=backend,
                num_requests=125, concurrency=25,
                max_tokens=256, prompt_set="conversation",
            ),
            # Phase 3: Full load (50 concurrent)
            BenchmarkConfig(
                model=model, backend=backend,
                num_requests=250, concurrency=50,
                max_tokens=256, prompt_set="conversation",
            ),
        ],
    )
