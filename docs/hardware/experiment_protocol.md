<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Hardware Experiment Protocol and Receipt

A training run, a conversion loss report or a
[target calibration](../api/conversion.md#target-fixed-point-calibration) is not
evidence of what a device does. That evidence is a run on the device, declared
before it happens and sealed afterwards. `sc_neurocore.hardware.experiment`
defines both halves. It never executes anything: execution is the operator's act
on the operator's device.

## Declare the protocol first

`resolve_protocol(document)` admits a protocol only when it could serve as
evidence, and refuses it with the offending field otherwise.

| Field | What it fixes |
|---|---|
| `opt_in` | Must be `true`: the operator explicitly authorises execution on the device |
| `declared_at` | ISO 8601 timestamp with a zone offset (for example `+00:00`); no run may start earlier |
| `operator` | `name` (required) and `contact` of whoever authorises and runs it |
| `device` | `vendor`, `model`, `serial` (required) and `firmware` |
| `image` | `kind` (`bitstream`, `firmware`, `container`, `binary`) and the `sha256` of what executes |
| `network_sha256`, `data_sha256` | The converted network (`converted_network.json`, conversion or target report) and the evaluation data |
| `samples` | Evaluation samples per measured run |
| `latency` | `start_event`, `end_event`, `clock`, `warmup_runs`, `measured_runs` and `includes_transport: true` |
| `power` | Optional: `instrument` (vendor, model, serial), `calibration` (certificate, calibrated_on, valid_until), `measurement_point`, `sample_rate_hz` |
| `criteria` | Preregistered `metric`/`threshold` pairs: `accuracy` (at least, in `[0, 1]`), `latency_p50_ms`, `latency_p95_ms`, `energy_per_inference_j` (at most) |

Latency is always measured from the host's request to the host receiving the
result, so transport is included; a protocol that does not say so is refused.
Warmup runs are declared, recorded and excluded from the percentiles. An energy
criterion requires a declared power instrument. `to_public_dict()` adds the
schema version and a `sha256` over every field; a stored protocol resubmitted
with a digest that no longer matches is refused.

## Seal what the run observed

`seal_receipt(protocol, observations)` takes raw observations: `started_at`,
`finished_at`, the `device_serial` and `image_sha256` seen at run time, one
latency per warmup and per measured run, the number of `correct` predictions,
optional `notes` and, when power was declared, `energy` with
`source: "instrument"`, the instrument's serial and joules per measured run.

It refuses a different device or image, a run that started before the protocol
was declared, run counts other than those declared, a missing energy
measurement when power was declared, energy when none was declared, energy from
another instrument or on a day outside its calibration's validity, and any
energy whose source is not the instrument. **Energy is never inferred from
operation counts or estimates.**

Accuracy, the nearest-rank p50 and p95 latencies, mean and maximum energy per
inference and the verdict of every preregistered criterion are computed from
the observations, never taken from the caller. The receipt embeds the full
protocol, its digest and a `sha256` over everything.

## Verify a receipt

`verify_receipt(document)` re-resolves the embedded protocol, checks its digest,
recomputes every derived figure and verdict from the raw observations and
compares the whole document, digest included. A receipt whose figures do not
follow from its observations is refused, so a stored receipt can be checked by
anyone holding it.

::: sc_neurocore.hardware.experiment
    options:
      show_root_heading: true
      members:
        - resolve_protocol
        - seal_receipt
        - verify_receipt
        - Protocol
        - HardwareExperimentError
