<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Taking an experiment out of the Studio

The Studio exports an experiment in two forms. A **script** reproduces the run
somewhere else. A **replay pack** additionally carries what the run produced, so
another installation can check that it still produces it.

Both are built from the resolved experiment — the same
`studio.experiment-spec.v1` the Studio runs — so the time step, the drive
protocol, the initial state, the parameter overrides and the randomness travel
with the export instead of being reconstructed from a guess.

## The script

`POST /api/codegen`, or the **Code** button, returns:

| Field | What it is |
|---|---|
| `script` | A standalone Python file that re-resolves the request and runs it |
| `oneliner` | The same experiment on one line, for a notebook |
| `replay_script` | A file that verifies a saved pack next to it |
| `experiment_sha256` | The digest the script checks before reporting a result |
| `request` | The pinned request the script carries |

The script is short on purpose:

```python
from sc_neurocore.studio.experiment_spec import resolve_experiment, run_experiment

REQUEST = {
    "current": 10.0,
    "dt": 0.05,
    "duration": 50.0,
    "name": "HodgkinHuxleyNeuron",
    "protocol": "step",
    "trial": "replay",
}

EXPERIMENT_SHA256 = "…"

spec = resolve_experiment(REQUEST)
if spec.experiment_sha256 != EXPERIMENT_SHA256:
    raise SystemExit("This installation resolves a different experiment …")

result = run_experiment(spec)
```

If the installed package would resolve a different experiment — a changed
model, descriptor, schema profile or numerical default — the script stops
instead of printing numbers under the exported experiment's name.

Editing a value in `REQUEST` is allowed and is the point of having it there;
the digest check then tells you that you are no longer running the exported
experiment.

## The replay pack

`POST /api/export/replay-pack`, or the **Replay pack** button, seals a
`studio.replay-pack.v1` document:

| Block | Contents |
|---|---|
| `request` | The re-resolvable request, with any drawn seed pinned and `trial` set to `replay` |
| `experiment` | The full public specification |
| `experiment_identity_sha256` | Digest of the scientific blocks only |
| `expectation` | Every spike event, a digest per state trace with endpoints and range, the initial and final state, the drive digest, the run statistics |
| `environment` | Package version, interpreter, NumPy, platform, machine |
| `runner` | How to replay it |

Run it anywhere the package is installed:

```bash
python -m sc_neurocore.studio.replay_pack pack.json
```

| Exit code | Meaning |
|---|---|
| 0 | The experiment reproduced (`match`, or `match-within-tolerance`) |
| 1 | It ran and did not reproduce; every difference is listed |
| 2 | It was refused before running; the reason is printed as JSON |

`--json` prints the whole outcome, `--tolerance` sets the largest absolute state
deviation still called a match (the default is exact), and
`--allow-runtime-drift` admits a different package, interpreter, NumPy or
platform and reports it rather than refusing.

### Identity versus runtime

`experiment_identity_sha256` covers the model revision, the numerical profile,
the step count, the parameters, the initial state, the protocol, the randomness
contract and the backend. It deliberately excludes the runtime block: a
different interpreter is drift to *report*, not a different experiment.

The digest describes the JSON value, not the Python object. A pack that a
browser saved has been through JavaScript's single number type, which narrows
`1000.0` to `1000`; that is the same experiment and the digest says so.

### What a replay refuses

| Stage | Refused when |
|---|---|
| `schema` | Unsupported pack version, a missing block, a document whose sealed digest does not describe its own specification, an unreadable or oversized file |
| `request` | The pack carries a request field this contract does not execute |
| `identity` | The sealed request no longer resolves here, or resolves to a different experiment |
| `revision` | The model's own revision differs |
| `runtime` | The sealed environment is not this one, and drift was not admitted |

Nothing runs until every refusal has been ruled out.

### What a replay compares

Spike events and the drive are compared exactly — a spike train is an
observable, not a rounding matter. State traces are compared by digest first
and, if the digests differ, by the largest deviation of their endpoints and
range against the tolerance; the outcome reports that number. Initial and final
states are compared per variable. A verdict of `match` means the digests agree;
`match-within-tolerance` means they did not and the deviation was accepted.

## Migrating from the earlier export

The earlier `/api/codegen` returned a script that constructed the model with its
default constructor, called `neuron.step(current=…)` and read `neuron.v`. Those
assumptions hold for part of the catalogue only, and where they did not the
script either failed to run or ran a different experiment from the one on
screen. The request body has changed accordingly:

- `mode` is now required and selects the schema. A body that does not say which
  kind of experiment it describes is rejected rather than assumed to be a model.
- A model request carries the model fields, an equation request the equation
  fields. Sending a null of the other branch is rejected; send only your own.
- `protocol`, `frequency_hz`, `seed` and `trial` are accepted and honoured. They
  were previously absent, so every export was a constant drive at the model's
  default time step.
- The response gained `replay_script`, `experiment_sha256` and `request`.

Scripts exported by the earlier endpoint keep working where they worked before;
they are not replay packs and carry no expectation, so nothing verifies them.
Re-export anything you rely on.
