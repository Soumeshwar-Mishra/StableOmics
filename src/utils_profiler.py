import os
import time
import json
import platform
import tracemalloc
import psutil
import numpy as np
import pandas as pd
from contextlib import contextmanager

class ResourceProfiler:
    """Profiles wall-clock runtime and peak resident memory (GB) for Table 4."""
    def __init__(self):
        self.records = []
        self.process = psutil.Process(os.getpid())

    @contextmanager
    def profile_task(self, task_name: str, n_genes: int, n_samples: str, replicate: int = 1):
        tracemalloc.start()
        rss_before = self.process.memory_info().rss / (1024 ** 3)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            _, peak_traced = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rss_after = self.process.memory_info().rss / (1024 ** 3)
            peak_ram_gb = max(rss_before, rss_after) + (peak_traced / (1024 ** 3))
            self.records.append({
                "Task": task_name,
                "Genes": n_genes,
                "Samples": str(n_samples),
                "Replicate": replicate,
                "Time_s": elapsed,
                "Peak_RAM_GB": peak_ram_gb
            })

    def get_summary_table(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame(self.records)
        summary = df.groupby(["Task", "Genes", "Samples"], as_index=False, sort=False).agg(
            Median_Time_s=("Time_s", "median"),
            Min_Time_s=("Time_s", "min"),
            Max_Time_s=("Time_s", "max"),
            Peak_RAM_GB=("Peak_RAM_GB", "max")
        )
        summary["Median time (s) [Range]"] = summary.apply(
            lambda r: f"{r['Median_Time_s']:.2f} ({r['Min_Time_s']:.2f}-{r['Max_Time_s']:.2f})", axis=1
        )
        summary["Peak RAM (GB)"] = summary["Peak_RAM_GB"].map(lambda x: f"{x:.3f}")
        return summary[["Task", "Genes", "Samples", "Median time (s) [Range]", "Peak RAM (GB)"]]

def save_run_manifest(config: dict, output_path: str):
    """Saves hardware, OS, software versions, and seeds to a JSON manifest."""
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "os": platform.platform(),
        "cpu_model": platform.processor(),
        "cpu_threads": os.cpu_count(),
        "system_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "python_version": platform.python_version(),
        "config": config
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
