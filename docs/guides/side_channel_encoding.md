<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore side-channel encoding guide -->

# Side-Channel Encoding

SC-NeuroCore includes analytic tooling for stochastic-computing bitstreams whose
switching activity should be inspected before deployment. The current workflow
generates deterministic activity-balanced streams, optional dummy streams,
analytic leakage-proxy reports, deploy/evidence manifests, and ROM-style HDL
hooks for precomputed bitstreams.

The evidence boundary is `analytic_simulation_only`.

## What The Evidence Means

The side-channel workflow can report:

- per-stream transition counts and transition rates;
- class-conditioned activity proxy values;
- baseline-versus-protected class-mean gap reduction;
- dummy-stream overhead ratio;
- protected bitstream count;
- total dummy streams inserted;
- generated HDL hook metadata.

The side-channel workflow does not provide:

- no physical power measurement;
- no physical thermal measurement;
- no DPA-resistance claim;
- no silicon-security claim.

Treat these artefacts as design-time evidence for stochastic-encoding review.
Board measurements, oscilloscope or power-rail traces, thermal captures,
place-and-route timing, and adversarial lab validation are separate evidence
classes and must be recorded separately.

## Python Workflow

Use the activity-balanced encoder for deterministic, probability-preserving
bitstream generation:

```python
from sc_neurocore.security import (
    ThermalSCEncodingConfig,
    encode_activity_balanced_probability,
)

record = encode_activity_balanced_probability(
    0.25,
    ThermalSCEncodingConfig(
        bitstream_length=64,
        seed=5,
        dummy_streams_per_record=1,
        max_dummy_overhead_ratio=1.0,
    ),
)
```

The returned record contains the payload bitstream, dummy bitstreams, realised
probability, switching-activity summary, dummy-stream count, seed domain, and
the `analytic_simulation_only` evidence boundary.

## Benchmark Report

Generate a canonical JSON benchmark report from the command line:

```bash
PYTHONPATH=src python tools/side_channel_benchmark.py \
  --output side_channel_benchmark.json \
  --probabilities 0.25,0.5 \
  --labels 0,1 \
  --bitstream-length 64 \
  --seed 5 \
  --dummy-streams-per-record 1 \
  --max-dummy-overhead-ratio 1.0
```

The report contains:

- `schema_version`;
- `evidence_boundary`;
- `deploy_manifest`;
- baseline class-activity proxy;
- protected class-activity proxy;
- max class-mean gap reduction;
- boundary notes;
- per-sample records.

The `deploy_manifest` section records the evidence class, benchmark artifact
path, security parameters, overhead measurements, and the same physical
evidence boundary notes.

## HDL Hook

Generate a ROM-style Verilog hook and hook manifest from the same protected
encoding configuration:

```bash
PYTHONPATH=src python tools/side_channel_hdl_emit.py \
  --verilog-output rtl/sc_side_channel_hook.v \
  --manifest-output rtl/sc_side_channel_hook_manifest.json \
  --module-name sc_side_channel_hook \
  --probability 0.25 \
  --bitstream-length 64 \
  --seed 5 \
  --dummy-streams-per-record 1 \
  --max-dummy-overhead-ratio 1.0
```

The emitted HDL exposes payload and dummy stream bits by `sample_index`. It is a
transport hook for precomputed analytic bitstreams. It is not a replacement for
hardware leakage measurement or silicon validation.

## Review Checklist

Before using a side-channel report in a deployment packet, check that:

- the report has `evidence_boundary: analytic_simulation_only`;
- `deploy_manifest.security_parameters` records bitstream length, seed,
  rotation stride, dummy stream count, and dummy overhead budget;
- `deploy_manifest.overhead_measurements` records the dummy overhead ratio and
  inserted dummy stream count;
- generated HDL hook manifests point to the Verilog file actually emitted;
- physical measurement claims are absent unless separate raw measurement
  artefacts exist.
