<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — SNN Studio user manual -->

# Run your first experiment

**Goal:** run one catalogue neuron, compare one deliberate input change, then
save and export enough information to return to it.
You need the local Studio interface; no FPGA tools or training dataset are
required. Follow [installation](installation.md) if it is not open yet.

## Read the workbench

The sidebar reports your workspace, design source, simulation, evidence health,
hardware readiness and available exports. Check these before starting:

- **Models** is selected and the catalogue has loaded.
- The selected model's name and parameter panel agree.
- Any capability warning names an action you intend to use. Missing FPGA tools
  do not prevent this single-neuron exercise.
- No earlier operation is still busy.

The guided flow marks the next available action. You can open **Trace** directly
without completing optional training or hardware steps.

## Select a simple neuron

1. Find **SCLapicqueLIFNeuron** in the catalogue and select it.
2. Read its state variables, units, timestep and parameter defaults.
3. Keep the model's parameter defaults for the first run.
4. Use a **constant** current protocol, current **1.2**, duration **50 ms** and
   timestep **0.1 ms**, where the model controls offer them.
5. Leave the seed empty for this deterministic exercise and use a replay trial.
6. Press **Space** outside a text field to run, or use the run action in the
   workbench. Wait until the simulation has finished.

The current is in the model's declared units. Do not interpret a generic
current slider as a universal biological amplitude across models. If this
model is absent from your build, report its installed version and choose a
model deliberately; a different model is a different experiment.

**Check the result:** the trace has a time axis, the current drive is present
and the run has a terminal success state. Inspect the reported spike count and
final state. A successful simulation may contain zero spikes; silence is a
result to investigate, not an automatic software failure.

## Read the trace

- Scroll over the trace to zoom in time, drag to pan and double-click to reset.
- Use the crosshair to inspect samples.
- Open **Data table** for a text/table representation of the displayed result.
- Check which variables are recorded and whether raw traces are included.

The plotted projection is bounded for display. Full-resolution raw traces and
state snapshots, when available, travel with the JSON result. A figure alone
is insufficient to reconstruct a run.

## Make one comparison

1. Save or export the baseline before changing it.
2. Change only the input current to **0** and rerun.
3. Compare the reported spike count, voltage trajectory and final state with
   the baseline. Explain any difference using this model's dynamics.
4. Restore current **1.2** and rerun before exporting the baseline experiment.

Changing an input withdraws completion of dependent analyses and hardware.
The previous trace can remain visible while a new request is pending or fails.
Read the run status before treating it as the current result.

## Save and return

1. Under **Projects**, choose **Save** and enter a new name such as `first-lif-run`.
2. Wait for the save acknowledgement; note the workspace revision if shown.
3. Use **Refresh**, then open the saved project.
4. Confirm the restored model, protocol, current, duration and timestep.
5. Run again. Loading settings is separate from generating a new result.

**Save session** under **Sessions** is a browser-local snapshot. It does not
create a server project. For the different storage and recovery guarantees,
read [Projects and evidence](projects-and-evidence.md).

## Export a result you can check

After the successful baseline run, use the toolbar:

1. **JSON** saves the returned run, including metadata and raw-data availability.
2. **CSV** saves full-resolution trace samples when raw export is available.
3. **Replay pack** saves the sealed request and expected outcome.
4. Confirm each file exists in your browser's download destination.

Replay the downloaded pack from the activated Python environment, using its
actual filename:

```bash
python -m sc_neurocore.studio.replay_pack pack.json --json
```

Exit **0** means the replay matched under the reported comparison rules;
**1** means it ran but differed; **2** means it was refused before running.
Read the report before changing runtime-drift or tolerance settings.
If state recording reports an incomplete contract, a matching replay does not
turn it into complete scientific-state coverage.

## Keyboard route

On the first visit, the tour keeps keyboard focus inside its dialog. Use
**Tab** or **Shift+Tab** to move between its controls, or **Escape** to dismiss
it. Dismissing or finishing the tour keeps it closed on your next visit.

Use **Tab** to reach controls and **Enter** to activate them. In the view
switcher, **Left/Right** change views; **Home/End** select the first/last enabled
view. Press **?** for shortcut help and **Esc** to close it. Number shortcuts
**1–5** select Trace, Phase plane, f-I curve, Bifurcation and Sensitivity when
applicable. Text fields and the equation editor retain their own editing keys.

Next: [Choose experiments and analyses](experiments.md).
