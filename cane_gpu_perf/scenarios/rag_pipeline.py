"""
Simulates a RAG (retrieval-augmented generation) pipeline.
- Long context inputs (4K-16K tokens of retrieved documents + question)
- Measures total pipeline latency including context assembly
- Tests context length scaling
"""

from cane_gpu_perf.scenarios.base import Scenario
from cane_gpu_perf.config import BenchmarkConfig


def rag_pipeline_scenario(model: str, backend: str) -> Scenario:
    return Scenario(
        name="RAG Pipeline",
        description=(
            "Simulates a production RAG system where retrieved documents "
            "(4K-16K context) are injected before the question. Tests how "
            "the model handles long-context generation and whether TTFT "
            "degrades with context length."
        ),
        latency_budget_ms=5000,
        success_criteria="TTFT p95 < 5000ms AND quality maintained across context lengths",
        configs=[
            # Short context (1K)
            BenchmarkConfig(
                model=model, backend=backend,
                num_requests=30, concurrency=4,
                max_tokens=512, prompt_set="rag",
                context_length=1024,
            ),
            # Medium context (4K)
            BenchmarkConfig(
                model=model, backend=backend,
                num_requests=30, concurrency=4,
                max_tokens=512, prompt_set="rag",
                context_length=4096,
            ),
            # Long context (16K)
            BenchmarkConfig(
                model=model, backend=backend,
                num_requests=20, concurrency=2,
                max_tokens=512, prompt_set="rag",
                context_length=16384,
            ),
        ],
    )
