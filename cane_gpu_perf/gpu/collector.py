"""GPU metrics collector using NVIDIA Management Library (NVML).

Samples GPU utilization, memory, power, temperature, and clock speeds
in a background thread during benchmarks. Gracefully returns None if
pynvml is unavailable or no NVIDIA GPU is present.
"""

import statistics
import threading
import time
from dataclasses import dataclass, field

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class GpuSnapshot:
    """Single point-in-time GPU measurement."""
    timestamp: float  # time.perf_counter()
    gpu_index: int
    utilization_gpu: int       # 0-100
    utilization_memory: int    # 0-100
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: int
    power_draw_w: float
    power_limit_w: float
    clock_sm_mhz: int
    clock_max_sm_mhz: int
    clock_mem_mhz: int
    pcie_tx_kbps: int
    pcie_rx_kbps: int


@dataclass
class GpuTimeSeries:
    """Aggregated GPU telemetry over a benchmark run."""
    gpu_index: int
    gpu_name: str
    gpu_uuid: str
    memory_total_mb: float
    num_samples: int
    duration_s: float
    snapshots: list[GpuSnapshot] = field(default_factory=list)

    # Computed aggregates (populated by finalize())
    utilization_mean: float = 0.0
    utilization_p95: float = 0.0
    utilization_min: float = 0.0

    memory_used_peak_mb: float = 0.0
    memory_used_mean_mb: float = 0.0

    temperature_peak_c: int = 0
    temperature_mean_c: float = 0.0

    power_draw_mean_w: float = 0.0
    power_draw_peak_w: float = 0.0
    power_limit_w: float = 0.0

    clock_sm_mean_mhz: float = 0.0
    clock_sm_min_mhz: int = 0
    clock_max_sm_mhz: int = 0
    clock_throttle_ratio: float = 0.0  # fraction of samples where clock < 90% of max

    pcie_tx_peak_kbps: int = 0
    pcie_rx_peak_kbps: int = 0

    def finalize(self):
        """Compute aggregates from raw snapshots."""
        if not self.snapshots:
            return

        utils = [s.utilization_gpu for s in self.snapshots]
        mems = [s.memory_used_mb for s in self.snapshots]
        temps = [s.temperature_c for s in self.snapshots]
        powers = [s.power_draw_w for s in self.snapshots]
        clocks = [s.clock_sm_mhz for s in self.snapshots]
        max_clocks = [s.clock_max_sm_mhz for s in self.snapshots]

        self.num_samples = len(self.snapshots)

        # Utilization
        self.utilization_mean = statistics.mean(utils) / 100.0
        self.utilization_min = min(utils) / 100.0
        sorted_utils = sorted(utils)
        idx_95 = min(int(len(sorted_utils) * 0.95), len(sorted_utils) - 1)
        self.utilization_p95 = sorted_utils[idx_95] / 100.0

        # Memory
        self.memory_used_peak_mb = max(mems)
        self.memory_used_mean_mb = statistics.mean(mems)

        # Temperature
        self.temperature_peak_c = max(temps)
        self.temperature_mean_c = statistics.mean(temps)

        # Power
        self.power_draw_mean_w = statistics.mean(powers)
        self.power_draw_peak_w = max(powers)
        self.power_limit_w = self.snapshots[0].power_limit_w

        # Clocks
        self.clock_sm_mean_mhz = statistics.mean(clocks)
        self.clock_sm_min_mhz = min(clocks)
        self.clock_max_sm_mhz = max(max_clocks) if max_clocks else 0
        if self.clock_max_sm_mhz > 0:
            throttled = sum(1 for c in clocks if c < self.clock_max_sm_mhz * 0.9)
            self.clock_throttle_ratio = throttled / len(clocks)

        # PCIe
        self.pcie_tx_peak_kbps = max(s.pcie_tx_kbps for s in self.snapshots)
        self.pcie_rx_peak_kbps = max(s.pcie_rx_kbps for s in self.snapshots)


class GpuCollector:
    """Collects GPU metrics in a background thread during benchmarks.

    Usage:
        collector = GpuCollector(interval_ms=100)
        collector.start()
        # ... run benchmark ...
        telemetry = collector.stop()  # list[GpuTimeSeries], one per GPU
    """

    def __init__(self, interval_ms: int = 100, gpu_indices: list[int] | None = None):
        self._interval_s = interval_ms / 1000.0
        self._gpu_indices = gpu_indices
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._snapshots: dict[int, list[GpuSnapshot]] = {}
        self._gpu_handles: dict[int, object] = {}
        self._gpu_names: dict[int, str] = {}
        self._gpu_uuids: dict[int, str] = {}
        self._gpu_mem_total: dict[int, float] = {}
        self._initialized = False
        self._start_time: float = 0.0

    def available(self) -> bool:
        """Check if GPU telemetry collection is possible."""
        if not NVML_AVAILABLE:
            return False
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            return count > 0
        except Exception:
            return False

    def start(self) -> bool:
        """Start collecting GPU metrics. Returns True if collection started."""
        if not NVML_AVAILABLE:
            return False

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                pynvml.nvmlShutdown()
                return False

            indices = self._gpu_indices or list(range(device_count))
            for idx in indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                self._gpu_handles[idx] = handle
                self._gpu_names[idx] = pynvml.nvmlDeviceGetName(handle)
                self._gpu_uuids[idx] = pynvml.nvmlDeviceGetUUID(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._gpu_mem_total[idx] = mem_info.total / (1024 * 1024)
                self._snapshots[idx] = []

            self._initialized = True
            self._stop_event.clear()
            self._start_time = time.perf_counter()
            self._thread = threading.Thread(target=self._collect_loop, daemon=True)
            self._thread.start()
            return True

        except Exception:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            return False

    def stop(self) -> list[GpuTimeSeries] | None:
        """Stop collection and return aggregated telemetry per GPU."""
        if not self._initialized:
            return None

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

        duration = time.perf_counter() - self._start_time
        results = []

        for idx in sorted(self._snapshots.keys()):
            ts = GpuTimeSeries(
                gpu_index=idx,
                gpu_name=self._gpu_names[idx],
                gpu_uuid=self._gpu_uuids[idx],
                memory_total_mb=self._gpu_mem_total[idx],
                num_samples=len(self._snapshots[idx]),
                duration_s=duration,
                snapshots=self._snapshots[idx],
            )
            ts.finalize()
            results.append(ts)

        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

        self._initialized = False
        return results

    def _collect_loop(self):
        """Background thread: sample all GPUs at the configured interval."""
        while not self._stop_event.is_set():
            t = time.perf_counter()
            for idx, handle in self._gpu_handles.items():
                try:
                    snapshot = self._sample_gpu(idx, handle, t)
                    self._snapshots[idx].append(snapshot)
                except Exception:
                    pass  # skip this sample on transient NVML errors
            elapsed = time.perf_counter() - t
            sleep_time = max(0, self._interval_s - elapsed)
            self._stop_event.wait(sleep_time)

    def _sample_gpu(self, idx: int, handle, timestamp: float) -> GpuSnapshot:
        """Take a single measurement from one GPU."""
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
        except pynvml.NVMLError:
            power = 0.0

        try:
            power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        except pynvml.NVMLError:
            power_limit = 0.0

        try:
            clock_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        except pynvml.NVMLError:
            clock_sm = 0

        try:
            clock_max_sm = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        except pynvml.NVMLError:
            clock_max_sm = 0

        try:
            clock_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except pynvml.NVMLError:
            clock_mem = 0

        try:
            pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
        except pynvml.NVMLError:
            pcie_tx = 0

        try:
            pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
        except pynvml.NVMLError:
            pcie_rx = 0

        return GpuSnapshot(
            timestamp=timestamp,
            gpu_index=idx,
            utilization_gpu=util.gpu,
            utilization_memory=util.memory,
            memory_used_mb=mem.used / (1024 * 1024),
            memory_total_mb=mem.total / (1024 * 1024),
            temperature_c=temp,
            power_draw_w=power,
            power_limit_w=power_limit,
            clock_sm_mhz=clock_sm,
            clock_max_sm_mhz=clock_max_sm,
            clock_mem_mhz=clock_mem,
            pcie_tx_kbps=pcie_tx,
            pcie_rx_kbps=pcie_rx,
        )
