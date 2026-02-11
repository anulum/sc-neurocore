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
3. **Build** the Rust engine:
   ```bash
   cd engine
   maturin develop --release
   ```
4. **Run tests** before making changes to establish a baseline:
   ```bash
   cargo test                           # Rust (90 tests)
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
  cd engine && cargo test && cd .. && python -m pytest tests/ -v
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

**High-value contributions:**
- New SIMD kernels (ARM SVE, RISC-V Vector)
- Additional HDC similarity metrics
- Jupyter notebook tutorials
- Performance benchmarks on new hardware
- Bug reports with reproducible test cases

**Please discuss first** (open an issue) before:
- Changing the public Python API
- Modifying the IR op set
- Adding new crate dependencies

## Submitting a Pull Request

1. Ensure all tests pass (`cargo test` + `pytest`)
2. Add a changelog entry if the change is user-visible
3. Open a PR against `main` with a clear description
4. Reference any related issues

## Licence

By contributing, you agree that your contributions will be licensed under the [GNU Affero General Public License v3.0](LICENSE). For commercial licensing enquiries, contact protoscience@anulum.li.
