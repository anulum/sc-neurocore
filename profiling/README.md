# Profiling

Tools and instructions for profiling SC-NeuroCore performance.

## Python Profiling

```bash
# cProfile
python -m cProfile -o profile.out -m pytest tests/ -x
python -m pstats profile.out

# line_profiler (install: pip install line-profiler)
kernprof -l -v benchmarks/benchmark_sc.py
```

## Rust Profiling

```bash
# Flamegraph (install: cargo install flamegraph)
cd engine && cargo flamegraph --bench sc_bench

# perf (Linux only)
cd engine && cargo bench --no-run
perf record --call-graph dwarf target/release/deps/sc_bench-*
perf report
```

Profiling scripts and results will be added to this directory as needed.
