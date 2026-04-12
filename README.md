# cane-gpu-perf

GPU inference benchmarking with opinionated diagnostics and deep hardware analysis.

## Install

```bash
pip install cane-perf
```

## Quick Start

```bash
# Single benchmark
cane-perf bench --model arcee-ai/trinity-large-thinking --backend openrouter --concurrency 8 --diagnose

# Deep GPU analysis (NVML telemetry, roofline model, power/thermal, prefill/decode)
cane-perf bench --model meta-llama/Llama-3-7b --backend vllm --concurrency 8 --deep

# With custom endpoint
cane-perf bench --model my-model --backend vllm --base-url http://localhost:8000/v1/chat/completions
```

## Deep GPU Analysis

The `--deep` flag enables hardware-level GPU profiling on top of standard HTTP-layer metrics.
Requires an NVIDIA GPU and `pip install cane-gpu-perf[gpu]`.

What it adds:

- **NVML telemetry** collected every 100ms during benchmarks: SM utilization, memory usage,
  power draw, temperature, clock speeds, PCIe throughput. Per-GPU when multi-GPU.
- **Prefill vs decode separation** with inter-token latency percentiles, prefill throughput
  (input tok/s), decode throughput (output tok/s), and time-in-phase breakdown.
- **Roofline model** classifying the workload as compute-bound, memory-bandwidth-bound, or
  under-utilized. Includes specs for 20+ GPUs (T4 through B200, RTX consumer cards).
- **Power and thermal efficiency**: tokens/watt, tokens/joule, electricity cost per 1M tokens,
  thermal headroom, clock throttle detection.
- **Multi-GPU topology**: NVLink vs PCIe interconnect detection, per-GPU utilization balance,
  straggler detection.

```bash
cane-perf bench --model meta-llama/Llama-3-7b --backend vllm --deep
```

Output:

```
Results:
  Requests:  100 total, 0 failed
  Latency:   p50=820ms  p95=1450ms  p99=2100ms
  TTFT:      p50=95ms   p95=180ms   p99=310ms
  Throughput: 142.3 tok/s aggregate, 28.5 tok/s mean per-request

Phase Analysis:
  Prefill:   1052 tok/s  (12% of request time)
  Decode:    31 tok/s  ITL p50=32ms p95=48ms p99=71ms

GPU 0: NVIDIA A100-SXM4-80GB
  Utilization    72%          min=45% p95=89%
  Memory         42.3 / 80.0 GB    peak=42.3GB mean=41.8GB
  Temperature    67C peak     mean=64C
  Power          287W mean    peak=312W limit=400W
  SM Clock       1410 MHz mean     min=1380 max=1410 MHz

Efficiency:
  Energy:    0.50 tok/W  0.0014 tok/J
  Power cost: $0.0561/1M tokens

## Findings

INFO: Memory-bandwidth-bound (62% of 2039 GB/s)
   Decode phase uses 62% of peak bandwidth. Expected for autoregressive generation.
   -> INT8 quantization: ~2x decode throughput. Speculative decoding: 2-3x.
   Expected impact: INT8: ~2x decode throughput. INT4: ~4x. Speculative decoding: 2-3x.

INFO: Prefill 1052 tok/s vs decode 31 tok/s (33.9x)
   Prefill processes tokens in parallel (compute-bound), decode is sequential (memory-bound).
   -> Focus optimization on whichever phase dominates your workload.
   Expected impact: Targeted optimization based on workload profile

INFO: Energy: 0.50 tok/W, 0.0014 tok/J
   Power draw: 287W mean / 312W peak (limit: 400W, 28% headroom).
   -> Compare across quantization levels: INT8 typically doubles tok/W.
   Expected impact: Quantization: ~2x energy efficiency for decode
```

## Workload Analysis

Run realistic workload scenarios and get actionable findings:

```bash
cane-perf analyze --model arcee-ai/trinity-large-thinking --backend vllm --scenario chatbot

# With deep GPU analysis across all scenario phases
cane-perf analyze --model meta-llama/Llama-3-7b --backend vllm --scenario chatbot --deep
```

Output:

```
## Findings

CRITICAL: GPU severely under-utilized (38%)
   GPU is idle more than half the time, waiting for data.
   -> Increase batch size or concurrency. Try --concurrency 8.
   Expected impact: 2-4x throughput improvement

WARNING: High TTFT variance (p99/p50 = 6.2x)
   Some requests take 6x longer than median.
   -> Investigate cold starts or KV cache eviction.
   Expected impact: More predictable user experience

INFO: Pareto-optimal configs: vllm, sglang
   Out of 4 configs, 2 are on the Pareto frontier.
   -> Choose vllm for latency, sglang for structured output.
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

## Diagnostic Categories

Standard diagnostics (always available with `--diagnose` or `--deep`):

| Category | What it checks |
|----------|---------------|
| `throughput` | GPU utilization, concurrency scaling, throughput ceiling |
| `latency` | TTFT, latency variance, p99/p50 ratio |
| `reliability` | Failure rate, error patterns |
| `memory` | GPU memory usage vs capacity |
| `config` | Concurrency settings, batching |
| `comparison` | Backend comparison, Pareto-optimal configs |
| `scaling` | Concurrency scaling efficiency |

Deep diagnostics (with `--deep`):

| Category | What it checks |
|----------|---------------|
| `roofline` | Compute-bound vs memory-bandwidth-bound classification |
| `phase_balance` | Prefill vs decode time split, inter-token latency |
| `thermal` | Clock throttling, temperature headroom |
| `efficiency` | Tokens/watt, energy cost, power headroom |
| `scaling` | Multi-GPU utilization balance, interconnect type |
| `memory_pressure` | KV cache growth, OOM risk under load |

## Architecture

```
cane_gpu_perf/
  config.py            # BenchmarkConfig, BenchmarkResult dataclasses
  utils/tokens.py      # tiktoken-based token counting
  bench/runner.py      # Benchmark runner (streaming HTTP, metrics, GPU collector)
  diagnose/engine.py   # DiagnoseEngine, opinionated findings (13 standard + 7 deep checks)
  scenarios/           # Workload scenarios (chatbot, rag, batch, code)
    base.py            # Scenario dataclass
    runner.py          # ScenarioRunner (multi-phase + SLA checks)
    *.py               # Individual scenario definitions
  gpu/                 # Deep GPU analysis (requires nvidia-ml-py)
    collector.py       # NVML telemetry collector (background thread, 100ms sampling)
    roofline.py        # Roofline model (20+ GPU specs, compute vs bandwidth classification)
    efficiency.py      # Power/thermal efficiency (tok/W, tok/J, cost, throttle detection)
    topology.py        # Multi-GPU topology (NVLink vs PCIe, utilization balance)
  cli/main.py          # CLI entry point (bench, analyze commands)
```

## Installation

```bash
pip install -e .

# With GPU telemetry support (requires NVIDIA GPU)
pip install -e ".[gpu]"
```
