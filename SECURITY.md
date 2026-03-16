# Security Policy

## Supported Versions

| Version | Supported          | Notes |
|---------|--------------------|-------|
| 3.12.x  | :white_check_mark: | Current stable (Rust engine + Python bridge) |
| 2.2.x   | :x:                | Superseded (pure-Python legacy) |
| < 2.0   | :x:                | Pre-release / unreleased |

Only the latest `3.x` patch receives security fixes. Upgrade with:

```bash
pip install --upgrade sc-neurocore-engine
```

## Reporting a Vulnerability

If you discover a security vulnerability in SC-NeuroCore, please report it
responsibly:

1. **GitHub Security Advisories** (preferred): [Report a vulnerability](https://github.com/anulum/sc-neurocore/security/advisories/new)
2. **Email:** protoscience@anulum.li — Subject: `[SECURITY] SC-NeuroCore — <brief description>`
3. **Do not** open a public GitHub issue for security vulnerabilities.

We will acknowledge receipt within 48 hours and aim to provide a fix within
7 days for critical issues.

## Scope

SC-NeuroCore is a neuromorphic computing simulation library. It does not handle
user authentication, financial data, or network services in its default
configuration. Security concerns are primarily:

- Malicious input files (JSON configs, NumPy `.npz`, ONNX models)
- Unsafe deserialization (pickle, NumPy load)
- Numerical overflow / denial of service via pathological inputs
- Native code memory safety (Rust engine via PyO3)
- HDL generation safety (SystemVerilog emitter input validation)
- Supply chain integrity (dependency audit)

## Hardening Measures in Place

### Rust Engine Safety
The v3 Rust engine uses safe Rust by default. All `unsafe` blocks are limited
to SIMD intrinsics with documented invariants. PyO3 boundary types enforce
correct Python ↔ Rust conversions.

### Input Validation
- Pickle allowlist restricts deserialization to known-safe classes
- Path traversal prevention on file I/O operations
- Array shape/dtype validation on all public API entry points

### Dependency Auditing
- **Rust:** `cargo audit` in CI
- **Python:** Minimal dependencies (`numpy`, `scipy`). No `pickle.load` of
  untrusted data in any production module.

## Known Limitations

- **No fuzzing harness yet.** Property-based testing via `proptest` (Rust) and
  `hypothesis` (Python) covers many input paths, but dedicated fuzzing has not
  been set up.
- **No third-party security audit.** The codebase has not been reviewed by an
  external security firm.
- **No CVE history.** No vulnerabilities have been reported to date.

Contributions to improve security coverage (fuzzing harnesses, static analysis
integration, audit reports) are welcome.
