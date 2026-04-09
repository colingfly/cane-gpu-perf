# cane-gpu-perf

GPU inference benchmarking with opinionated diagnostics. Don't just measure — diagnose.

## Install

```bash
pip install -e .
```

## Quick Start

```bash
# Single benchmark
cane-perf bench --model arcee-ai/trinity-large-thinking --backend openrouter --concurrency 8 --diagnose

# With custom endpoint
cane-perf bench --model my-model --backend vllm --base-url http://localhost:8000/v1/chat/completions
```

## Workload Analysis

Run realistic workload scenarios and get actionable findings:

```bash
cane-perf analyze --model arcee-ai/trinity-large-thinking --backend vllm --scenario chatbot
```

Output:

```
## Findings

🔴 GPU severely under-utilized (38%)
   GPU is idle more than half the time, waiting for data.
   → Increase batch size or concurrency. Try --concurrency 8.
   Expected impact: 2-4x throughput improvement

🟡 High TTFT variance (p99/p50 = 6.2x)
   Some requests take 6x longer than median.
   → Investigate cold starts or KV cache eviction.
   Expected impact: More predictable user experience

🔵 Pareto-optimal configs: vllm, sglang
   Out of 4 configs, 2 are on the Pareto frontier.
   → Choose vllm for latency, sglang for structured output.
   Expected impact: Eliminate suboptimal configurations
```

Available scenarios: `chatbot`, `rag`, `batch`, `code`

## Scenarios

| Scenario | What it simulates | Key metric |
|----------|-------------------|------------|
| `chatbot` | 50 concurrent chat users, 3 load phases | TTFT p95 < 1500ms |
| `rag` | Long-context RAG pipeline (1K-16K tokens) | TTFT p95 < 5000ms |
| `batch` | Offline batch processing, concurrency sweep | Max aggregate tok/s |
| `code` | Coding assistant (autocomplete + full gen) | TTFT < 300ms (autocomplete) |

## Architecture

```
cane_gpu_perf/
├── config.py          # BenchmarkConfig, BenchmarkResult dataclasses
├── utils/tokens.py    # tiktoken-based token counting
├── bench/runner.py    # Benchmark runner (streaming HTTP, metrics collection)
├── diagnose/engine.py # DiagnoseEngine — opinionated findings
├── scenarios/         # Workload scenarios (chatbot, rag, batch, code)
│   ├── base.py        # Scenario dataclass
│   ├── runner.py      # ScenarioRunner (multi-phase + SLA checks)
│   └── *.py           # Individual scenario definitions
└── cli/main.py        # CLI entry point
```
