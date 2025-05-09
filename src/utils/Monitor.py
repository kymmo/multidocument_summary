import threading
import time
import psutil
from contextlib import ContextDecorator
from typing import Optional
try:
     from pynvml import (
          nvmlInit,
          nvmlShutdown,
          nvmlDeviceGetCount,
          nvmlDeviceGetHandleByIndex,
          nvmlDeviceGetMemoryInfo,
          nvmlDeviceGetUtilizationRates,
          NVMLError
     )
     HAVE_NVML = True
except ImportError:
     HAVE_NVML = False

class ResourceMonitor(ContextDecorator):
     def __init__(self, interval: float = 60.0, label: str = "MON"):
          self.interval = interval
          self.label = label
          self._stop_event = threading.Event()
          self._thread: Optional[threading.Thread] = None
          self.gpu_handles = []
          self._nvml_initialized = False

          if HAVE_NVML:
               try:
                    nvmlInit()
                    self._nvml_initialized = True
                    if (gpu_count := nvmlDeviceGetCount()) > 0:
                         self.gpu_handles = [nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)]
               except NVMLError as e:
                    print(f"NVML Init Error: {str(e)}")
                    self.gpu_handles = []

     def _monitor(self):
          while not self._stop_event.is_set():
               try:
                    # CPU / RAM / Swap
                    cpu_pct = psutil.cpu_percent(interval=0.1)
                    vmem = psutil.virtual_memory()
                    swap = psutil.swap_memory()

                    gpu_stats = []
                    if self._nvml_initialized:
                         for idx, h in enumerate(self.gpu_handles):
                              try:
                                   mem = nvmlDeviceGetMemoryInfo(h)
                                   util = nvmlDeviceGetUtilizationRates(h)
                                   gpu_stats.append(
                                        f"GPU{idx} util={util.gpu}% "
                                        f"mem={mem.used/2**30:.2f}/{mem.total/2**30:.2f} GB"
                                   )
                              except NVMLError as e:
                                   gpu_stats.append(f"GPU{idx} Error: {str(e)}")

                    log_msg = (
                         f"[{self.label}] Memory Usage"
                         f"CPU={cpu_pct:.1f}% | "
                         f"RAM={vmem.percent:.1f}% ({vmem.used/2**30:.2f}/{vmem.total/2**30:.2f} GB) | "
                         f"SWAP={swap.percent:.1f}% ({swap.used/2**30:.2f}/{swap.total/2**30:.2f} GB)"
                    )
                    if gpu_stats:
                         log_msg += " | " + " | ".join(gpu_stats)

                    print(log_msg)

               except Exception as e:
                    print(f"Monitoring error: {str(e)}")

               self._stop_event.wait(self.interval)

     def __enter__(self):
          if not self._thread or not self._thread.is_alive():
               self._stop_event.clear()
               self._thread = threading.Thread(target=self._monitor, daemon=True)
               self._thread.start()
          return self

     def __exit__(self, exc_type, exc, tb):
          self.stop()
          if self._nvml_initialized:
               try:
                    nvmlShutdown()
               except NVMLError:
                    pass
          return False

     def stop(self):
          self._stop_event.set()
          if self._thread and self._thread.is_alive():
               self._thread.join(timeout=2.0)
          self._thread = None