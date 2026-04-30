# Security Policy

## Supported Versions

| Version | Supported          | Notes |
|---------|--------------------|-------|
| 3.13.x  | :white_check_mark: | Current stable (Rust engine + Python bridge) |
| 2.2.x   | :x:                | Superseded (pure-Python legacy) |
| < 2.0   | :x:                | Pre-release / unreleased |

Only the latest `3.x` patch receives security fixes. Upgrade with:

```bash
pip install --upgrade sc-neurocore
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
- **SBOM/release artefacts:** run the offline audit locally with
  `python tools/supply_chain_audit.py`. Use `--strict` before releases to fail
  on SBOM metadata drift as well as hard errors such as un-hashed release
  requirements.

## Known Limitations

- **No fuzzing harness yet.** Property-based testing via `proptest` (Rust) and
  `hypothesis` (Python) covers many input paths, but dedicated fuzzing has not
  been set up.
- **No third-party security audit.** The codebase has not been reviewed by an
  external security firm.
- **No CVE history.** No vulnerabilities have been reported to date.

Contributions to improve security coverage (fuzzing harnesses, static analysis
integration, audit reports) are welcome.

## 2026-04-30 Hardening Status Update

The older limitation above is now partially superseded. The repository has
Python property-based fuzz coverage for high-risk structured inputs, including
bitstream/IR parsing, NIR import, model-zoo `.npz` inputs, chip-spec JSON,
optimizer evidence JSON, SCPN datastream JSON, Studio graph JSON,
equation-to-MLIR lowering, HDL-source lowering, and transfer-checkpoint inputs.

Current release-blocking security checks:

```bash
pytest tests/test_fuzz_*.py tests/test_hypothesis_properties.py -q
python tools/supply_chain_audit.py --strict
bandit -r src/sc_neurocore/ -c pyproject.toml -q
```

Remaining hardening work:

- Add dedicated Rust `cargo-fuzz` targets for native parsers and PyO3 boundary
  adapters.
- Keep expanding Python Hypothesis coverage for malicious `.nir`, `.npz`, JSON,
  and pathological bitstream-length cases.
- Complete an external third-party security review before making audited
  security claims.
- No paid bug-bounty program is active yet. Until a scoped bounty is funded,
  coordinated disclosure through GitHub Security Advisories and the security
  email above is the official process.
