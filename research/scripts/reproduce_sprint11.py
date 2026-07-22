import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.benchmark_runner import run_baselines_benchmark

if __name__ == "__main__":
    print("=== Reproducing Sprint 11 Benchmark Runner ===")
    results = run_baselines_benchmark()
    print("\nBenchmark Results:")
    print(results)
