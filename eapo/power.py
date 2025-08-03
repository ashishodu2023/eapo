import time
import threading
from contextlib import contextmanager

try:
    import pynvml
    _NVML = True
except Exception:
    _NVML = False

class PowerLogger:
    def __init__(self, interval_ms: int = 50, device_index: int = 0):
        self.interval = interval_ms / 1000.0
        self.device_index = device_index
        self.samples_watts = []
        self._run = False
        self._thread = None
        self._handle = None
        self.available = False
        if _NVML:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                self.available = True
            except Exception:
                self.available = False

    def _sample_loop(self):
        while self._run:
            try:
                p_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self.samples_watts.append(p_mw / 1000.0)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.samples_watts.clear()
        self._run = True
        if self.available:
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._run = False
        if self._thread:
            self._thread.join()

    def energy_joules(self, elapsed_s: float) -> float:
        if not self.available or not self.samples_watts:
            return float("nan")
        mean_watts = sum(self.samples_watts) / len(self.samples_watts)
        return mean_watts * elapsed_s

@contextmanager
def measure_energy(nvml_interval_ms: int = 50, device_index: int = 0):
    pl = PowerLogger(nvml_interval_ms, device_index)
    t0 = time.perf_counter()
    pl.start()
    try:
        yield pl
    finally:
        pl.stop()
        elapsed = time.perf_counter() - t0
        pl.elapsed = elapsed

