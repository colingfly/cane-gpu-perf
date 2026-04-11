"""Multi-GPU topology detection and scaling analysis.

Detects NVLink vs PCIe interconnect, per-GPU utilization balance,
and interconnect saturation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class GpuLink:
    """Interconnect between two GPUs."""
    gpu_a: int
    gpu_b: int
    link_type: str  # "nvlink", "pcie", "single", "unknown"


@dataclass
class TopologyReport:
    """Multi-GPU topology and balance analysis."""
    gpu_count: int
    gpu_names: list[str]
    links: list[GpuLink] = field(default_factory=list)
    interconnect_type: str = "unknown"  # "nvlink", "pcie", "mixed", "single"

    # Per-GPU utilization balance (from telemetry)
    utilization_per_gpu: list[float] = field(default_factory=list)  # mean util 0-1
    utilization_stdev: float = 0.0
    utilization_imbalance: float = 0.0  # max - min

    # Per-GPU memory
    memory_used_per_gpu_gb: list[float] = field(default_factory=list)
    memory_total_per_gpu_gb: list[float] = field(default_factory=list)

    # PCIe bandwidth
    pcie_tx_peak_per_gpu_kbps: list[int] = field(default_factory=list)
    pcie_rx_peak_per_gpu_kbps: list[int] = field(default_factory=list)


def detect_topology() -> TopologyReport | None:
    """Detect multi-GPU topology using NVML.

    Returns None if NVML unavailable or fewer than 1 GPU found.
    """
    if not NVML_AVAILABLE:
        return None

    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()

        if count == 0:
            pynvml.nvmlShutdown()
            return None

        gpu_names = []
        mem_total = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_names.append(pynvml.nvmlDeviceGetName(handle))
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_total.append(mem.total / (1024 ** 3))

        links = []
        has_nvlink = False
        has_pcie = False

        if count > 1:
            for i in range(count):
                for j in range(i + 1, count):
                    link_type = _detect_link_type(i, j)
                    links.append(GpuLink(gpu_a=i, gpu_b=j, link_type=link_type))
                    if link_type == "nvlink":
                        has_nvlink = True
                    elif link_type == "pcie":
                        has_pcie = True

        if count == 1:
            interconnect = "single"
        elif has_nvlink and has_pcie:
            interconnect = "mixed"
        elif has_nvlink:
            interconnect = "nvlink"
        elif has_pcie:
            interconnect = "pcie"
        else:
            interconnect = "unknown"

        pynvml.nvmlShutdown()

        return TopologyReport(
            gpu_count=count,
            gpu_names=gpu_names,
            links=links,
            interconnect_type=interconnect,
            memory_total_per_gpu_gb=mem_total,
        )

    except Exception:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return None


def _detect_link_type(gpu_a: int, gpu_b: int) -> str:
    """Detect the link type between two GPUs."""
    try:
        handle_a = pynvml.nvmlDeviceGetHandleByIndex(gpu_a)
        handle_b = pynvml.nvmlDeviceGetHandleByIndex(gpu_b)

        # Check NVLink
        try:
            topo = pynvml.nvmlDeviceGetTopologyCommonAncestor(handle_a, handle_b)
            # NVML topology levels:
            # NVML_TOPOLOGY_INTERNAL = 0
            # NVML_TOPOLOGY_SINGLE = 10
            # NVML_TOPOLOGY_MULTIPLE = 20
            # NVML_TOPOLOGY_HOSTBRIDGE = 30
            # NVML_TOPOLOGY_NODE = 40
            # NVML_TOPOLOGY_SYSTEM = 50
            if topo <= 20:  # SINGLE or MULTIPLE (NVLink connected)
                return "nvlink"
            else:
                return "pcie"
        except pynvml.NVMLError:
            pass

        # Fallback: check NVLink status directly
        try:
            for link_idx in range(6):  # up to 6 NVLink connections
                try:
                    state = pynvml.nvmlDeviceGetNvLinkState(handle_a, link_idx)
                    if state:
                        remote = pynvml.nvmlDeviceGetNvLinkRemotePciInfo(handle_a, link_idx)
                        # If we can read NVLink state, NVLink exists
                        return "nvlink"
                except pynvml.NVMLError:
                    continue
        except Exception:
            pass

        return "pcie"

    except Exception:
        return "unknown"


def analyze_gpu_balance(gpu_telemetry_list: list) -> TopologyReport | None:
    """Analyze utilization balance across multiple GPUs.

    Args:
        gpu_telemetry_list: list[GpuTimeSeries] from GpuCollector
    """
    if not gpu_telemetry_list or len(gpu_telemetry_list) < 1:
        return None

    report = detect_topology()
    if report is None:
        return None

    import statistics

    utils = [ts.utilization_mean for ts in gpu_telemetry_list]
    report.utilization_per_gpu = utils
    if len(utils) > 1:
        report.utilization_stdev = statistics.stdev(utils)
        report.utilization_imbalance = max(utils) - min(utils)

    report.memory_used_per_gpu_gb = [ts.memory_used_peak_mb / 1024.0 for ts in gpu_telemetry_list]

    report.pcie_tx_peak_per_gpu_kbps = [ts.pcie_tx_peak_kbps for ts in gpu_telemetry_list]
    report.pcie_rx_peak_per_gpu_kbps = [ts.pcie_rx_peak_kbps for ts in gpu_telemetry_list]

    return report
