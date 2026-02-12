# Contributing to SC-NeuroCore

CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
Contact us: www.anulum.li  protoscience@anulum.li

Thank you for your interest in SC-NeuroCore. Contributions are welcome under the following guidelines.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork and create a feature branch:
   ```bash
   git clone https://github.com/<your-user>/sc-neurocore.git
   cd sc-neurocore
   git checkout -b feature/your-feature
   ```
3. **Install** the engine (choose one):
   ```bash
   # Option A: from PyPI (for Python-only contributions)
   pip install sc-neurocore-engine

   # Option B: build from source (for Rust/Python contributions)
   cd bridge
   pip install maturin
   maturin develop --release
   ```
4. **Run tests** before making changes to establish a baseline:
   ```bash
   cd engine && cargo test            # Rust (90 tests)
   cd .. && python -m pytest tests/ -v  # Python (49 tests)
   ```

## Development Guidelines

### Code Style

- **Rust**: Follow standard `rustfmt` formatting. Run `cargo fmt` before committing.
- **Python**: Follow PEP 8. Use type hints for public APIs.
- **Copyright header**: Every source file must include the standard header:
  ```
  CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
  Contact us: www.anulum.li  protoscience@anulum.li
  ORCID: https://orcid.org/0009-0009-3560-0851
  License: GNU AFFERO GENERAL PUBLIC LICENSE v3
  Commercial Licensing: Available
  ```

### Testing

- All new Rust code must include tests in `engine/tests/` or inline `#[cfg(test)]` modules.
- All new Python APIs must have pytest coverage in `tests/`.
- SIMD paths must include portable fallback — never assume AVX-512 or AVX2 availability.
- Run the full suite before submitting:
  ```bash
  cd engine && cargo fmt --check && cargo test && cd .. && python -m pytest tests/ -v
  ```

### Commit Messages

Follow conventional commit format:
```
feat(scope): short description
fix(scope): short description
docs(scope): short description
```

Examples:
```
feat(bitstream): add rotate_left method for HDC permutation
fix(simd): correct AVX2 popcount for non-aligned buffers
docs(readme): update benchmark table for v3.8
```

## What to Contribute

### Good First Issues

These are great starting points for new contributors:

- **Add a Jupyter notebook tutorial** — convert an existing `examples/*.py` to `.ipynb` with explanations
- **Expand troubleshooting docs** — encountered a build issue? Document the fix
- **Add HDC similarity metrics** — implement cosine or Jaccard on `BitStreamTensor`
- **Benchmark on new hardware** — run `examples/03_benchmark_report.py` and share results (ARM, Apple M-series, AMD)
- **Improve error messages** — find a cryptic `assert!` in Rust and replace with a descriptive message

### High-Value Contributions

- New SIMD kernels (ARM SVE, RISC-V Vector)
- Additional HDC/VSA operations (thin binding, resonator networks)
- Jupyter notebook tutorials with `%timeit` benchmarks
- Performance benchmarks on new hardware
- Bug reports with reproducible test cases
- Documentation improvements

### Please Discuss First

Open an issue before working on:
- Changes to the public Python API
- Modifications to the IR op set
- Adding new crate dependencies
- Architectural changes

## Submitting a Pull Request

1. Ensure all tests pass (`cargo fmt --check && cargo test` + `pytest`)
2. Add a changelog entry if the change is user-visible
3. Open a PR against `main` with a clear description
4. Reference any related issues

### PR Checklist

- [ ] `cargo fmt --check` passes
- [ ] `cargo test` passes (all Rust tests)
- [ ] `python -m pytest tests/ -v` passes (all Python tests)
- [ ] New public APIs have docstrings/doc comments
- [ ] SIMD code has portable fallback path
- [ ] Copyright header present in new files

## Project Structure

```
sc-neurocore/
├── engine/             # Rust crate (PyO3 + maturin)
│   ├── src/            # Core modules (bitstream, simd, layer, neuron, ir, scpn)
│   ├── tests/          # Rust integration tests
│   └── benches/        # Criterion benchmarks
├── bridge/             # Python package (sc_neurocore_engine)
│   └── sc_neurocore_engine/
│       ├── __init__.py # Public API exports
│       ├── hdc.py      # HDCVector class
│       └── petri_net.py # PetriNetEngine class
├── tests/              # Python pytest suite
├── examples/           # Runnable demo scripts
├── notebooks/          # Jupyter notebooks
├── cosim/              # Hardware co-simulation (Verilator)
└── hdl/                # SystemVerilog reference designs
```

## Licence

By contributing, you agree that your contributions will be licensed under the [GNU Affero General Public License v3.0](LICENSE). For commercial licensing enquiries, contact protoscience@anulum.li.
