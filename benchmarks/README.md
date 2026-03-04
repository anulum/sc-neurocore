# Benchmarks

Performance benchmarks for the SC-NeuroCore framework.

## Scripts

| Script | Description |
|--------|-------------|
| `bench_v2_vs_v3.py` | Compare v2 (pure-Python) vs v3 (Rust engine) performance |
| `benchmark_advanced_modules.py` | Benchmark advanced module operations |
| `benchmark_sc.py` | Core stochastic computing primitives |
| `benchmark_suite.py` | Full 14-benchmark suite across 5 categories |

## Running

```bash
# Quick run
python benchmarks/benchmark_suite.py

# Full run (10x iterations) with markdown output
python benchmarks/benchmark_suite.py --full --markdown

# v2 vs v3 comparison
python benchmarks/bench_v2_vs_v3.py
```

## Rust Benchmarks

Criterion benchmarks are defined in `engine/benches/` and run via:

```bash
cd engine && cargo bench
```
