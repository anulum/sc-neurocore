# Security Hardening

This page tracks current security hardening evidence and the remaining roadmap
work. It complements `SECURITY.md`; it does not replace coordinated disclosure
through GitHub Security Advisories or the project security email.

## 2026-04-30 Status

### Property-based fuzzing

Python fuzz coverage exists for the current high-risk structured input surface:

| Surface | Test file |
|---|---|
| Bitstream and IR parsing | `tests/test_fuzz_bitstream_ir.py` |
| NIR import | `tests/test_fuzz_nir_import_inputs.py` |
| Model-zoo `.npz` input handling | `tests/test_fuzz_model_zoo_npz_inputs.py` |
| Chip-spec JSON | `tests/test_fuzz_chip_spec_json_inputs.py` |
| Optimizer evidence JSON | `tests/test_fuzz_optimizer_evidence_json_inputs.py` |
| SCPN datastream JSON | `tests/test_fuzz_scpn_datastream_json_inputs.py` |
| Studio graph JSON | `tests/test_fuzz_studio_graph_inputs.py` |
| Equation-to-MLIR lowering | `tests/test_fuzz_equation_mlir_lowering_inputs.py` |
| HDL-source lowering | `tests/test_fuzz_hdl_source_lowering_inputs.py` |
| Transfer-checkpoint input handling | `tests/test_fuzz_transfer_checkpoint_inputs.py` |
| Fuzz/property suites | Dedicated `tests/test_fuzz_*.py` files |

Run the focused Python fuzz/property suite:

```bash
pytest tests/test_fuzz_*.py -q
```

Tagged releases run a bounded Hypothesis subset through the release security
sweep:

```bash
python -m pip install --require-hashes -r requirements/fuzz.txt
python tools/security_scan/release_security_sweep.py \
  --output-dir security/ci-security-packet \
  --include-fuzz \
  --fuzz-max-total-time 300
```

Dedicated Rust `cargo-fuzz` targets now start under `fuzz/` for the SC IR
parser and core bitstream operations. Rust parser and SIMD safety evidence also
comes from the existing Rust tests, property sweeps, and the documented
`unsafe` invariants at the PyO3/SIMD boundary.

Run the initial native fuzz targets:

```bash
cargo install cargo-fuzz --version 0.13.2
cargo fuzz run ir_parser
cargo fuzz run bitstream_ops
```

### Supply-chain audit

The release supply-chain check is:

```bash
python tools/supply_chain_audit.py --strict
```

Use `--strict` before publishing release artifacts so SBOM metadata drift,
unhashed release requirements, and dependency-audit failures block the release.
The tag workflow runs this check after Syft writes
`security/ci-security-packet/security/sbom.cdx.json` and records
`security/ci-security-packet/security/supply_chain_audit.json`.

### Static analysis

Run Bandit on Python sources:

```bash
bandit -r src/sc_neurocore/ -c pyproject.toml -q
```

Tagged releases also run Semgrep from the repository-owned `.semgrep.yml`
policy instead of registry-hosted rule packs:

```bash
python -m pip install --no-deps --require-hashes -r requirements/semgrep.txt
python tools/security_scan/run_semgrep_scanners.py \
  --output-dir security/ci-security-packet
```

CI also carries the broader quality gates documented in the release process.

### Bug-bounty status

No paid bug-bounty program is active as of 2026-04-30. The active process is
coordinated disclosure through GitHub Security Advisories or the security email
listed in `SECURITY.md`.

Before announcing a bounty, define:

1. In-scope assets and versions.
2. Reward tiers for parser crashes, arbitrary code execution, data exfiltration,
   dependency-confusion, and denial-of-service findings.
3. Exclusions for speculative model-output concerns that do not cross a security
   boundary.
4. Disclosure timelines and public-credit policy.

## Remaining Roadmap

- Extend `cargo-fuzz` harnesses from IR parser and bitstream operations into
  PyO3 boundary adapters and additional native import/decoder paths.
- Add crash-corpus preservation for minimized Python Hypothesis examples.
- Extend malicious-input coverage for `.nir`, `.npz`, JSON, and pathological
  bitstream lengths as new import paths land.
- Run an external third-party security audit before publishing audited-security
  claims.
- Fund and publish a scoped bug-bounty program once the disclosure budget and
  triage rota are ready.
