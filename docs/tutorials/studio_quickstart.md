# Tutorial: Visual SNN Studio Quickstart

Run a catalogue neuron, compare an input change, then save and reproduce the
experiment. No training dataset, Rust compiler or FPGA tool is required for
this exercise. Studio is a development preview; advanced actions depend on
the current model and deployment capabilities.

## 1. Open Studio

From your activated environment with the `studio` extra installed:

```bash
sc-neurocore studio
```

Open `http://127.0.0.1:8001/studios/sc-neurocore/` if the browser does not open.
If API documentation appears instead, check the
[frontend installation](../studio/manual/installation.md).

## 2. Run a small experiment

1. Choose **Models**, then **SCLapicqueLIFNeuron**.
2. Keep model parameters at their defaults. Use constant current **1.2**,
   duration **50 ms** and timestep **0.1 ms** in the available controls.
3. Leave the seed empty for this deterministic run.
4. Run with **Space** outside a text field or the workbench's run action.
5. Wait for success, inspect **Trace**, spike count, final state and **Data table**.

Read units from the model. Zero spikes can be a valid result. An older trace
remaining on screen does not mean a failed rerun succeeded.
The [full first-experiment procedure](../studio/manual/first-experiment.md)
explains the controls and checks.

## 3. Compare one change

Export the baseline with **JSON**, change only the current to **0**, and run
again. Compare the trajectories and reported spikes. Restore the baseline
current and rerun before exporting that experiment.

## 4. Save and reproduce

1. Use **Projects → Save** with a new name, then **Refresh** and reopen it.
2. Check the restored inputs; a project save keeps settings, not every result.
3. Download **JSON**, **CSV** and **Replay pack**, and check the saved files.
4. Replay the pack with the activated environment and its actual filename:

```bash
python -m sc_neurocore.studio.replay_pack pack.json --json
```

Exit 0 means a match under the reported rules, 1 an executed mismatch, and 2 a
refusal before execution. **Sessions → Save** is separate browser-local storage.
See [Projects and evidence](../studio/manual/projects-and-evidence.md).

## 5. Choose your next workflow

| Goal | Continue here |
|---|---|
| Custom ODEs, sweeps, precision and comparisons | [Experiments and analysis](../studio/manual/experiments.md) |
| Populations, projections, training, candidates or fitting | [Networks and training](../studio/manual/networks-and-training.md) |
| RTL, parity, FPGA target profiles and synthesis | [Hardware workflow](../studio/manual/hardware.md) |
| Login, jobs, recovery or a failed operation | [Troubleshooting](../studio/manual/troubleshooting.md) |

The current Rust **Build IR** adapter constructs a fixed LIF graph and is not a
translation of arbitrary entered equations. Structural IR verification does
not establish source fidelity. Use the hardware guide's separate catalogue
and direct-Python compiler procedures and retain their qualification limits.

[Complete user manual](../studio/manual/index.md) ·
[Technical Studio guide](../guides/studio.md)
