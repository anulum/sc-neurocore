# Safety evidence

The sc_neurocore.safety_cert package organises caller-supplied safety evidence.
It provides traceability, FMEDA arithmetic, formal-property manifests,
formula-based timing analysis, checklist bookkeeping, deterministic report
hashes, and atomic local package materialisation.

It does **not** certify a product, grant regulatory approval, reproduce
normative standards text, or replace an independent assessor. Clause labels and
cross-standard links are curated navigation aids. Applicability and conformity
decisions require licensed standards and qualified review.

## Fail-closed contract

The package never promotes a repository path, template, or module name into
evidence automatically.

- Requirements remain open until the caller supplies implementation references.
- A proven formal property becomes a verification link, but a requirement is
  verified only when it also has explicit implementation evidence.
- FMEDA stays not assessed until the caller supplies FailureMode records and
  their FIT/DC provenance.
- WCET stays not assessed until all four network dimensions and clock frequency
  are supplied.
- Checklist rows start not_addressed. Caller-supplied evidence moves a row only
  to partial; a conformity decision is outside this library.
- Timestamps are aware ISO-8601 values. An explicit generated_at value makes
  package output reproducible.

The legacy FMEDA.add_sc_standard_modes helper is retained for compatibility,
but it now requires acknowledge_synthetic_profile=True. Its rates are synthetic
test assumptions and are not safety-case evidence.

## Module ownership

The historical sc_neurocore.safety_cert.safety_cert import path is a thin
compatibility facade. Implementation ownership is split by responsibility:

| Module | Responsibility |
|---|---|
| standards.py | SafetyStandard, SILLevel, ASILLevel, and the explicitly non-normative SIL/ASIL label crosswalk |
| traceability.py | Requirement records, implementation/verification links, coverage, and concrete artifact reporting |
| failure_analysis.py | FailureMode validation, FMEDA arithmetic, residual risk, SFF/DC screens, and reliability metrics |
| formal_evidence.py | Full-field formal-evidence hashing, reports, proof completeness, and gap detection |
| timing_analysis.py | Formula-derived pipeline cycle counts and clock conversion |
| compliance.py | Fail-closed checklist templates, IEC 62304 label screen, and curated clause navigation |
| certification.py | Evidence-package assembly, content hashing, atomic materialisation, and manual templates |
| fault_tolerance.py | Non-normative common-cause and HFT screening helpers |
| change_impact.py | Change records and re-verification bookkeeping |
| evidence.py | Path-safe evidence manifests, full SHA-256 digests, and on-disk verification |
| safety_cert.py | Stable import and pickle facade only |

The graph is acyclic. Public class pickle paths remain
sc_neurocore.safety_cert.safety_cert so existing serialized records remain
loadable.

## End-to-end package

The generator accepts its historical positional inputs and adds explicit
keyword-only evidence inputs:

~~~python
from pathlib import Path

from sc_neurocore.safety_cert import (
    CertificationGenerator,
    EvidenceBag,
    FailureCategory,
    FailureMode,
    FormalProperty,
    SILLevel,
    SafetyStandard,
)

properties = [
    FormalProperty(
        prop_id="P-SAFE-001",
        module="neuron",
        description="Accumulator remains in range",
        property_type="assert",
        status="proven",
        engine="SymbiYosys 2.4.0",
        depth=32,
        sby_file="formal/neuron.sby",
    )
]

failure_modes = [
    FailureMode(
        fm_id="FM-NEURON-001",
        component="neuron",
        description="Accumulator register upset",
        category=FailureCategory.DANGEROUS_DETECTED,
        failure_rate_fit=12.5,
        diagnostic_coverage=0.95,
        mitigation="Parity monitor; verification report E-17",
    )
]

package = CertificationGenerator().generate(
    SafetyStandard.IEC_61508,
    SILLevel.SIL_2,
    ["neuron"],
    properties,
    {
        "bitstream_length": 256,
        "num_inputs": 8,
        "num_neurons": 16,
        "clock_mhz": 100.0,
    },
    implementation_evidence={"neuron": ["rtl/neuron.sv"]},
    failure_modes=failure_modes,
    checklist_evidence={"7.4.2": "evidence/formal-review.md"},
    generated_at="2026-07-12T18:30:00+00:00",
)

destination = package.write(Path("build/safety-evidence"))

bag = EvidenceBag()
bag.add_from_package(package)
assert bag.verify(destination)
~~~

CertificationPackage.write creates a new destination only. It writes all
artifacts into a private sibling directory, flushes them, computes real
SHA-256 digests, emits manifest.json, and renames the complete directory.
Existing destinations are never intentionally overwritten. A stale package ID,
empty report, partial write, symlinked evidence file, missing file, or digest
mismatch fails closed.

The materialised directory contains:

| File | Content |
|---|---|
| traceability_matrix.md | Requirement status plus concrete implementation and verification references |
| fmeda_report.md | Caller-supplied FIT/DC arithmetic and provenance warning, or not assessed |
| formal_proof_cert.md | Formal-property table, full evidence SHA-256, and compatible 32-character certificate ID |
| wcet_analysis.md | Formula-derived cycles and timing inputs, or not assessed |
| compliance_checklist.md | Evidence bookkeeping without a conformity claim |
| manifest.json | Schema, package ID, full package digest, and per-file byte counts/SHA-256 digests |

The 32-character package_hash and certificate_hash fields remain compatibility
identifiers. Integrity-sensitive consumers should use content_sha256 and the
full manifest digests.

## Core analyses

### Traceability

TraceabilityMatrix stores uniquely identified Requirement objects. Coverage is
verified requirements divided by all requirements; implemented-only rows no
longer inflate the numerator. Reports list both counts and the actual artifact
references.

### FMEDA and reliability

FMEDA performs arithmetic over explicit FailureMode records. The caller owns
the provenance and applicability of every FIT rate and diagnostic-coverage
value. max_achievable_sil, ReliabilityMetrics.pfh_sil, CCFAnalysis, HFTAssessment,
and ProofTestCoverage.dc_to_sil are legacy screening labels, not SIL
determinations.

### Formal evidence

FormalProofCertificate hashes every material property field:

- property ID and module;
- description and property type;
- status, engine, and depth;
- SymbiYosys file reference;
- certificate tool version.

Changing any field changes content_sha256. Duplicate IDs and corrupted
in-memory records are rejected before hashing.

### Timing

WCETAnalyzer is a reviewable formula model for pipeline cycle counts. It is not
a static-timing-analysis report, silicon measurement, operating-corner proof,
or hardware certification result.

### Checklists and mappings

ComplianceChecklist contains curated clause labels for IEC 61508, ISO 26262,
FDA Class III, DO-254, and EN 50129. Template artifact locations are not copied
into evidence fields. CrossStandardMapper reports curated relationships and
does not assert normative equivalence.

## Runtime monitor boundary

SafetyMonitor and SafetyLimits are separate runtime-monitor APIs. The existing
Rust, Julia, Go, and Mojo SafetyMonitor surfaces remain outside this refactor.
Certification report assembly is Python orchestration, not a numerical kernel.
The former Rust/Julia/Go/Mojo safety_cert report-generator files were generated
stubs: Julia and Mojo did not parse, Go contained empty functions, and Rust
returned constants. They were removed instead of being presented as
acceleration.

## Verification and benchmark evidence

The focused safety cohort executes unit, boundary, architecture, pickle,
materialisation, tamper, public-API, and industrial integration tests. All
touched production modules are required to reach 100 percent statement and
branch coverage and pass strict MyPy, Ruff, public docstring policy, and SPDX
checks.

benchmarks/bench_safety_certification.py compares the parent and modular source
trees using interleaved fresh interpreters. The committed local result records
30 samples per variant, source SHA-256 values, CPU affinity, load, raw samples,
and medians. It is local regression context only; it is not a publishable
throughput claim.

## Public API

The package continues to export the established 32 evidence symbols plus
SafetyLimits and SafetyMonitor. Direct imports from the historical facade and
top-level package retain object identity.
