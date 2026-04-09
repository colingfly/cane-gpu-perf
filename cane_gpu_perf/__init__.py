"""cane-gpu-perf: GPU inference benchmarking with opinionated diagnostics."""

__version__ = "0.1.0"

from cane_gpu_perf.config import BenchmarkConfig, BenchmarkResult
from cane_gpu_perf.diagnose.engine import DiagnoseEngine, Finding, format_findings

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "DiagnoseEngine",
    "Finding",
    "format_findings",
]
