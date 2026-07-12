# Building a safety-evidence package

This guide shows how to assemble reviewable evidence with
sc_neurocore.safety_cert. The output is an engineering evidence bundle, not a
certificate, regulatory submission, approval, or claim of conformity.

Use the applicable licensed standards and qualified independent reviewers for
all normative decisions.

## 1. Establish provenance before coding

For every value you plan to include, record:

- source artifact and immutable revision;
- responsible owner and review status;
- tool name/version and command;
- execution environment and timestamp;
- whether the value is measured, derived, assumed, or illustrative;
- uncertainty, operating conditions, and known limitations.

Do not substitute a repository path for evidence. A file name proves only that
a file was named; it does not prove content, applicability, independence, or
approval.

## 2. Define requirements

Requirement IDs must be unique and stable. Implementation and verification are
separate link sets.

~~~python
from sc_neurocore.safety_cert import (
    Requirement,
    SILLevel,
    SafetyStandard,
    TraceabilityMatrix,
)

matrix = TraceabilityMatrix()
matrix.add_requirement(
    Requirement(
        req_id="REQ-SAFE-001",
        description="The accumulator shall remain within its configured range",
        standard=SafetyStandard.IEC_61508,
        sil_level=SILLevel.SIL_2,
    )
)

matrix.link_implementation("REQ-SAFE-001", "rtl/neuron.sv@sha256:...")
matrix.link_verification("REQ-SAFE-001", "formal/neuron-proof.json@sha256:...")

assert matrix.coverage == 1.0
~~~

An implementation-only row is implemented, not verified. A verification-only
link leaves the requirement open because no implementation artifact is known.

## 3. Add formal-property evidence

FormalProperty records are evidence metadata. A status of proven is meaningful
only with the referenced tool output, configuration, assumptions, and version.

~~~python
from sc_neurocore.safety_cert import FormalProofCertificate, FormalProperty

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

certificate = FormalProofCertificate(
    properties=properties,
    tool_version="SymbiYosys 2.4.0",
)
certificate.compute_hash(generated_at="2026-07-12T18:30:00+00:00")

print(certificate.content_sha256())
~~~

The full digest covers every material property field and the certificate tool
version. The shorter certificate_hash remains only as a compatibility ID.

## 4. Supply FMEDA data explicitly

Create FailureMode entries only from reviewed source data. Record the model,
device/process/environment scope, diagnostic assumptions, and evidence behind
each mitigation.

~~~python
from sc_neurocore.safety_cert import FailureCategory, FailureMode, FMEDA

mode = FailureMode(
    fm_id="FM-NEURON-001",
    component="neuron",
    description="Accumulator register upset",
    category=FailureCategory.DANGEROUS_DETECTED,
    failure_rate_fit=12.5,
    diagnostic_coverage=0.95,
    mitigation="Parity monitor; reviewed evidence E-17",
)

fmeda = FMEDA()
fmeda.add_failure_mode(mode)
print(fmeda.generate_report())
~~~

FMEDA.add_sc_standard_modes is a legacy demo helper. It requires explicit
acknowledge_synthetic_profile=True and must not be used as measured FIT/DC
evidence.

## 5. Make timing assumptions visible

The built-in WCETAnalyzer is a deterministic formula model. Its inputs are not
silicon measurements or synthesis timing.

~~~python
from sc_neurocore.safety_cert import WCETAnalyzer

path = WCETAnalyzer.analyze(
    bitstream_length=256,
    num_inputs=8,
    num_neurons=16,
)
print(path.total_cycles)
print(path.wcet_ns(clock_mhz=100.0))
~~~

For safety use, attach the implementation revision, clock source, operating
corners, tool report, margin policy, and measurement or static-timing method.

## 6. Attach checklist evidence

Checklist generation is fail-closed:

~~~python
from sc_neurocore.safety_cert import ComplianceChecklist, SafetyStandard

empty = ComplianceChecklist.generate(SafetyStandard.IEC_61508)
assert all(item.status == "not_addressed" for item in empty)

partial = ComplianceChecklist.generate(
    SafetyStandard.IEC_61508,
    evidence={"7.4.2": "evidence/formal-review.md"},
)
assert partial[0].status == "partial"
~~~

The library never marks a row compliant. The supplied mapping records where
evidence can be reviewed; it is not a conformity decision.

## 7. Assemble and materialise

CertificationGenerator keeps the established positional API and adds explicit
keyword-only evidence inputs.

~~~python
from pathlib import Path

from sc_neurocore.safety_cert import (
    CertificationGenerator,
    SILLevel,
    SafetyStandard,
)

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
    implementation_evidence={"neuron": ["rtl/neuron.sv@sha256:..."]},
    failure_modes=[mode],
    checklist_evidence={"7.4.2": "evidence/formal-review.md"},
    generated_at="2026-07-12T18:30:00+00:00",
)

output = package.write(Path("build/safety-evidence"))
~~~

write requires a new destination. It creates five Markdown reports and one JSON
manifest. The manifest contains the package schema, full content digest,
compatibility package ID, standard/target labels, timestamp, and SHA-256/byte
count for every report.

If the package is mutated after its ID was calculated, write refuses the stale
ID. Empty reports, existing destinations, and partial I/O also fail.

## 8. Verify materialised evidence

~~~python
from sc_neurocore.safety_cert import EvidenceBag

bag = EvidenceBag()
bag.add_from_package(package)

assert bag.verify(output)
print(bag.manifest())
~~~

EvidenceItem accepts only normalised relative POSIX paths. Verification rejects
missing/unhashed files, symlinks, path traversal, and digest mismatches.

For long-term evidence storage, additionally capture the manifest in an
append-only or signed record system under your organisation's controls.

## 9. Interpret legacy screens conservatively

Several APIs predate the fail-closed package writer:

- SIL_TO_ASIL;
- ProofTestCoverage.dc_to_sil;
- FMEDA.max_achievable_sil;
- ReliabilityMetrics.pfh_sil;
- CCFAnalysis and HFTAssessment;
- IEC62304Assessment.from_sil.

They return coarse screening labels for compatibility. No single metric or
crosswalk establishes SIL, ASIL, software class, regulatory applicability, or
certification.

## 10. Reproducible verification

Use a fixed aware generated_at timestamp when comparing packages. Pin tool
versions and hash every external artifact referenced in the reports.

The focused implementation checks cover:

- strict runtime validation and corrupted-state boundaries;
- exact public/facade ownership and historical pickle paths;
- fail-closed generator behavior;
- full-field evidence hashing;
- atomic materialisation and I/O cleanup;
- tamper and missing-file detection;
- absence of false certification runtime mirrors;
- 100 percent statement and branch coverage for the touched production graph;
- a source-bound parent/candidate local benchmark.

See the [API reference](../api/safety_cert.md) for the module map and complete
contract notes.
