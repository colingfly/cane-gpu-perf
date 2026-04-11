"""Power and thermal efficiency analysis for GPU inference.

Computes tokens-per-watt, energy cost estimates, and detects
thermal throttling from GPU telemetry data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EfficiencyReport:
    """Power and thermal efficiency metrics."""
    tokens_per_watt: float
    tokens_per_joule: float
    total_energy_j: float
    total_tokens: int

    # Cost estimate
    power_cost_per_1m_tokens: float | None  # USD at given electricity rate
    electricity_rate_kwh: float  # USD/kWh used for estimate

    # Thermal
    temperature_peak_c: int
    temperature_mean_c: float
    thermal_headroom_c: float   # throttle_temp - peak_temp
    throttle_temp_c: int        # assumed or detected throttle point

    # Clock stability
    clock_throttle_ratio: float  # fraction of time running below 90% max clock
    clock_mean_mhz: float
    clock_max_mhz: int
    clock_min_mhz: int

    # Power
    power_mean_w: float
    power_peak_w: float
    power_limit_w: float
    power_headroom_pct: float   # (limit - mean) / limit


def analyze_efficiency(gpu_telemetry, aggregate_tps: float, total_tokens: int,
                       duration_s: float, electricity_rate_kwh: float = 0.10) -> EfficiencyReport | None:
    """Compute efficiency metrics from GPU telemetry.

    Args:
        gpu_telemetry: GpuTimeSeries for the primary GPU
        aggregate_tps: Overall tokens per second
        total_tokens: Total tokens generated
        duration_s: Benchmark duration in seconds
        electricity_rate_kwh: Electricity cost in USD per kWh (default $0.10)
    """
    if gpu_telemetry is None:
        return None

    ts = gpu_telemetry
    if ts.power_draw_mean_w <= 0:
        return None

    # Tokens per watt
    tokens_per_watt = aggregate_tps / ts.power_draw_mean_w if ts.power_draw_mean_w > 0 else 0

    # Energy
    total_energy_j = ts.power_draw_mean_w * duration_s
    total_energy_kwh = total_energy_j / 3_600_000

    # Tokens per joule
    tokens_per_joule = total_tokens / total_energy_j if total_energy_j > 0 else 0

    # Cost per 1M tokens
    if tokens_per_joule > 0:
        energy_per_token_kwh = 1.0 / (tokens_per_joule * 3_600_000)
        power_cost_per_1m_tokens = energy_per_token_kwh * 1_000_000 * electricity_rate_kwh
    else:
        power_cost_per_1m_tokens = None

    # Thermal throttle temp heuristic
    # Data center GPUs (A100, H100): ~83C
    # Consumer GPUs (RTX): ~83-90C
    throttle_temp = 83
    thermal_headroom = throttle_temp - ts.temperature_peak_c

    # Power headroom
    power_headroom_pct = ((ts.power_limit_w - ts.power_draw_mean_w) / ts.power_limit_w * 100
                          if ts.power_limit_w > 0 else 0)

    return EfficiencyReport(
        tokens_per_watt=tokens_per_watt,
        tokens_per_joule=tokens_per_joule,
        total_energy_j=total_energy_j,
        total_tokens=total_tokens,
        power_cost_per_1m_tokens=power_cost_per_1m_tokens,
        electricity_rate_kwh=electricity_rate_kwh,
        temperature_peak_c=ts.temperature_peak_c,
        temperature_mean_c=ts.temperature_mean_c,
        thermal_headroom_c=thermal_headroom,
        throttle_temp_c=throttle_temp,
        clock_throttle_ratio=ts.clock_throttle_ratio,
        clock_mean_mhz=ts.clock_sm_mean_mhz,
        clock_max_mhz=ts.clock_max_sm_mhz,
        clock_min_mhz=ts.clock_sm_min_mhz,
        power_mean_w=ts.power_draw_mean_w,
        power_peak_w=ts.power_draw_peak_w,
        power_limit_w=ts.power_limit_w,
        power_headroom_pct=power_headroom_pct,
    )
