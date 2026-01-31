import time
import statistics
from typing import List

class LatencyTracker:
    def __init__(self, max_samples=1000):
        self.samples: List[float] = []
        self.max_samples = max_samples
        self.start_times = {}

    def tick(self, tag: str):
        self.start_times[tag] = time.perf_counter()

    def tock(self, tag: str) -> float:
        if tag not in self.start_times:
            return 0.0
        
        duration = (time.perf_counter() - self.start_times[tag]) * 1000.0 # ms
        self.samples.append(duration)
        
        # Keep window size
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)
            
        return duration

    def get_stats(self):
        if not self.samples:
            return {"avg": 0, "max": 0, "min": 0, "jitter": 0}
        
        avg = statistics.mean(self.samples)
        return {
            "avg_ms": avg,
            "max_ms": max(self.samples),
            "min_ms": min(self.samples),
            "jitter_ms": statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0,
            "count": len(self.samples)
        }

    def print_stats(self):
        stats = self.get_stats()
        print(f"Latency Stats (N={stats['count']}):")
        print(f"  Avg: {stats['avg_ms']:.2f} ms")
        print(f"  Max: {stats['max_ms']:.2f} ms")
        print(f"  Jitter: {stats['jitter_ms']:.2f} ms")
