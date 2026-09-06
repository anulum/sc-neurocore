<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Checking an evidence pack without the Studio

An exported pack used to carry a digest of each file it contained. That proves
the bytes were not edited after they were written. It does not say whether the
analysis beside a simulation was computed from it, whether the two agree about
which model ran, or whether the exporter — who wrote both the evidence and the
verdict — was right.

A pack now carries a **receipt** on each piece of evidence and a **chain
document** recording what the exporter concluded. Both can be rechecked from
disk by anyone who receives the pack.

## The receipt

Every receipt has the same shape, whatever lane produced it:

| Field | What it says |
|---|---|
| `receipt_id` | `<lane>.<first 32 characters of the seal>` — the artefact's identity |
| `lane` | Evidence class: `simulation`, `analysis`, `training`, `compile`, … |
| `status` | Terminal status of the action that produced it |
| `binding` | `produced` or `exported` — see below |
| `seal_sha256` | Digest of the payload without its own receipt |
| `scope` | Identity it was produced under: experiment digest, model class, descriptor and schema digests, numerical profile |
| `depends_on` | Inputs it rests on, named by an identity value rather than by position |
| `produced_at_utc` | When it was produced |

`binding` is the distinction that matters when reading someone else's pack. A
`produced` receipt was written by the surface that ran the work, so it attests
what actually ran. An `exported` receipt was written while assembling the pack
from payloads an operator supplied, so it attests only what the exporter
received. Both are honest; they are not the same claim.

## The seal survives the browser

A payload leaving the Studio passes through the operator's browser before it
comes back to be exported. `JSON.stringify` writes `1` where Python writes
`1.0`, `-70` where Python writes `-70.0`, and `1e-7` where Python writes
`1e-07`. Measured on a real `AdExNeuron` run before this contract existed: the
recorded digest was `771d8e51…` and the identical payload re-digested to
`813d3005…` after that round trip. Nothing had changed except the rendering, and
no verifier could have existed while that was true.

`studio.evidence-seal.v1` encodes values rather than one runtime's rendering of
them, and the server and the browser implement it identically — one shared set
of vectors is checked from both sides. Anything that would not survive the trip
intact, such as a non-finite number or an integer no double holds exactly, is
refused rather than silently altered.

## The verdicts

`verify_evidence_chain` re-seals every subject and resolves every declared input
inside the pack:

| Verdict | What it means |
|---|---|
| `verified` | The payload matches its receipt and rests on inputs that do too |
| `tampered` | The payload no longer seals to the digest its receipt records |
| `scope_mismatch` | It and a resolved input disagree about the model or the profile |
| `stale` | It predates the input it claims, or descends from something that failed a check |
| `missing_dependency` | An input it names is not in this pack |
| `unsealed` | It came from a build that wrote no receipt |

An export is refused when anything is `tampered`, `scope_mismatch` or `stale` —
the pack would contradict itself. `missing_dependency` and `unsealed` are
reported instead: exporting a run without its inputs is incomplete, not
dishonest, and the chain document records `verified` and `complete` separately
so the difference is visible.

## Rechecking a pack you were sent

```bash
PYTHONPATH=src:. python tools/studio_evidence_verify.py path/to/pack
```

The tool reads the pack's own `evidence/manifest.json`, recomputes every file
digest, re-seals every subject against its receipt, re-resolves the dependency
graph, and then compares its finding against the verdict the pack records. It
exits non-zero when a file is missing or altered, when a subject no longer
matches its seal, or when the recorded verdict and the recomputed one disagree.
`--json <path>` writes the same finding as a sorted document.

A pack with no chain document is not taken on trust either: the tool reports
that the pack records no verdict of its own.

## What a receipt does not claim

- It does not sign anything. A receipt establishes that a pack is internally
  consistent and unaltered since export, not who produced it.
- `depends_on` is declared only where a derivation genuinely exists — a restored
  checkpoint on the training job that wrote it, an attached checkpoint on the
  restore, a guided-flow attestation on the run it attests. A simulation rests
  on nothing else in a pack and declares nothing; an invented edge would make a
  chain look checked where it was not.
- A payload from a build older than `studio.evidence-receipt.v1` reads as
  `unsealed`. That is a boundary the report names, not a check that passed.
