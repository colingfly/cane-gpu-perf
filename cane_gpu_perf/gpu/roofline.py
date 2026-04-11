"""Roofline model analysis for GPU inference workloads.

Determines whether a workload is compute-bound or memory-bandwidth-bound
by comparing achieved throughput against the GPU's theoretical ceilings.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


# Known GPU specs: cuda_cores, tensor_cores, memory_bandwidth_gbps
# Used as fallback when NVML can't provide all details.
GPU_SPECS = {
    # Data center
    "A100": {"fp16_tflops": 312, "mem_bw_gbps": 2039, "hbm": True},
    "A100-SXM": {"fp16_tflops": 312, "mem_bw_gbps": 2039, "hbm": True},
    "A100-PCIE": {"fp16_tflops": 312, "mem_bw_gbps": 1935, "hbm": True},
    "H100": {"fp16_tflops": 990, "mem_bw_gbps": 3350, "hbm": True},
    "H100-SXM": {"fp16_tflops": 990, "mem_bw_gbps": 3350, "hbm": True},
    "H100-PCIE": {"fp16_tflops": 756, "mem_bw_gbps": 2039, "hbm": True},
    "H200": {"fp16_tflops": 990, "mem_bw_gbps": 4800, "hbm": True},
    "B200": {"fp16_tflops": 2250, "mem_bw_gbps": 8000, "hbm": True},
    "L40": {"fp16_tflops": 181, "mem_bw_gbps": 864, "hbm": False},
    "L40S": {"fp16_tflops": 366, "mem_bw_gbps": 864, "hbm": False},
    "L4": {"fp16_tflops": 121, "mem_bw_gbps": 300, "hbm": False},
    "A10": {"fp16_tflops": 125, "mem_bw_gbps": 600, "hbm": False},
    "A10G": {"fp16_tflops": 125, "mem_bw_gbps": 600, "hbm": False},
    "T4": {"fp16_tflops": 65, "mem_bw_gbps": 300, "hbm": False},
    "V100": {"fp16_tflops": 125, "mem_bw_gbps": 900, "hbm": True},
    "V100-SXM2": {"fp16_tflops": 125, "mem_bw_gbps": 900, "hbm": True},
    # Consumer
    "RTX 4090": {"fp16_tflops": 165, "mem_bw_gbps": 1008, "hbm": False},
    "RTX 4080": {"fp16_tflops": 97, "mem_bw_gbps": 717, "hbm": False},
    "RTX 4070 Ti": {"fp16_tflops": 81, "mem_bw_gbps": 504, "hbm": False},
    "RTX 3090": {"fp16_tflops": 71, "mem_bw_gbps": 936, "hbm": False},
    "RTX 3080": {"fp16_tflops": 47, "mem_bw_gbps": 760, "hbm": False},
    "RTX 5090": {"fp16_tflops": 209, "mem_bw_gbps": 1792, "hbm": False},
}


@dataclass
class RooflineResult:
    """Result of roofline analysis."""
    gpu_name: str
    peak_fp16_tflops: float
    peak_mem_bw_gbps: float
    ridge_point: float          # FLOPs/byte where compute ceiling meets bandwidth ceiling

    # Achieved metrics
    achieved_compute_pct: float  # % of peak compute utilized
    achieved_bandwidth_pct: float  # % of peak memory bandwidth utilized

    # Classification
    classification: str  # "compute-bound", "memory-bound", "under-utilized"
    bottleneck_phase: str  # "prefill", "decode", or "both"

    # Decode-specific (autoregressive generation is almost always memory-bound)
    decode_bytes_per_token: float  # estimated bytes read from memory per output token
    decode_bandwidth_utilization: float  # what fraction of bandwidth is used during decode


def lookup_gpu_specs(gpu_name: str) -> dict | None:
    """Look up known GPU specs by name, with fuzzy matching."""
    # Exact match
    if gpu_name in GPU_SPECS:
        return GPU_SPECS[gpu_name]

    # Substring match
    name_upper = gpu_name.upper()
    for key, specs in GPU_SPECS.items():
        if key.upper() in name_upper or key.upper().replace(" ", "") in name_upper.replace(" ", ""):
            return specs

    return None


def estimate_model_bytes(model_name: str) -> float | None:
    """Rough estimate of model weight size in bytes from the model name.

    This is a heuristic -- looks for parameter count hints in the name
    (e.g., '7b', '13b', '70b') and assumes fp16 (2 bytes per param).
    """
    import re
    name_lower = model_name.lower()

    # Look for patterns like "7b", "13b", "70b", "405b"
    match = re.search(r'(\d+\.?\d*)\s*b(?:illion)?', name_lower)
    if match:
        params_b = float(match.group(1))
        # Assume fp16 = 2 bytes per parameter
        return params_b * 1e9 * 2

    return None


def analyze_roofline(gpu_name: str, model_name: str,
                     prefill_throughput_tps: float,
                     decode_throughput_tps: float,
                     prompt_tokens_mean: float,
                     gpu_utilization_mean: float | None = None,
                     mem_bw_utilization: float | None = None) -> RooflineResult | None:
    """Perform roofline analysis for the given GPU and workload.

    Args:
        gpu_name: GPU name string from NVML
        model_name: Model name (used to estimate parameter count)
        prefill_throughput_tps: Prefill tokens/sec (input tokens / TTFT)
        decode_throughput_tps: Decode tokens/sec
        prompt_tokens_mean: Mean prompt length in tokens
        gpu_utilization_mean: GPU SM utilization 0-1 (from NVML)
        mem_bw_utilization: Memory bandwidth utilization 0-1 (from NVML memory util)
    """
    specs = lookup_gpu_specs(gpu_name)
    if specs is None:
        return None

    peak_fp16_tflops = specs["fp16_tflops"]
    peak_mem_bw_gbps = specs["mem_bw_gbps"]

    # Ridge point: where compute ceiling meets bandwidth ceiling
    # Units: FLOPs per byte
    ridge_point = (peak_fp16_tflops * 1e12) / (peak_mem_bw_gbps * 1e9)

    model_bytes = estimate_model_bytes(model_name)

    # Estimate bytes read per decode token:
    # For autoregressive decoding, each token requires reading all model weights
    # plus KV cache. Weight read dominates for large models.
    if model_bytes:
        decode_bytes_per_token = model_bytes  # simplified: full weight read per token
    else:
        # Conservative default: assume 14B fp16 model
        decode_bytes_per_token = 14e9 * 2

    # Decode bandwidth utilization
    # bytes/sec consumed = decode_tps * bytes_per_token
    decode_bw_bytes_per_sec = decode_throughput_tps * decode_bytes_per_token
    decode_bw_gbps = decode_bw_bytes_per_sec / 1e9
    decode_bandwidth_utilization = decode_bw_gbps / peak_mem_bw_gbps if peak_mem_bw_gbps > 0 else 0

    # Compute utilization estimate
    # For prefill: ~2 * params * tokens FLOPs (forward pass)
    # We use GPU utilization from NVML as a proxy
    achieved_compute_pct = (gpu_utilization_mean or 0) * 100
    achieved_bandwidth_pct = decode_bandwidth_utilization * 100

    # Also factor in NVML memory utilization if available
    if mem_bw_utilization is not None:
        # NVML memory utilization is a better direct measure
        achieved_bandwidth_pct = max(achieved_bandwidth_pct, mem_bw_utilization * 100)

    # Classification
    if achieved_compute_pct < 30 and achieved_bandwidth_pct < 30:
        classification = "under-utilized"
    elif achieved_bandwidth_pct > achieved_compute_pct:
        classification = "memory-bound"
    else:
        classification = "compute-bound"

    # Determine bottleneck phase
    # Prefill is typically compute-bound, decode is typically memory-bound
    if prefill_throughput_tps > 0 and decode_throughput_tps > 0:
        # If decode throughput is the limiting factor
        bottleneck_phase = "decode" if decode_throughput_tps < prefill_throughput_tps * 0.5 else "prefill"
    else:
        bottleneck_phase = "both"

    return RooflineResult(
        gpu_name=gpu_name,
        peak_fp16_tflops=peak_fp16_tflops,
        peak_mem_bw_gbps=peak_mem_bw_gbps,
        ridge_point=ridge_point,
        achieved_compute_pct=achieved_compute_pct,
        achieved_bandwidth_pct=achieved_bandwidth_pct,
        classification=classification,
        bottleneck_phase=bottleneck_phase,
        decode_bytes_per_token=decode_bytes_per_token,
        decode_bandwidth_utilization=decode_bandwidth_utilization,
    )
