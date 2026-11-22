# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio graph-to-target pipeline

"""Studio graph-to-target pipeline.

:func:`run_pipeline` takes a canvas graph through validation and simulation. It
stops there: it reports a hardware result only if it can lower the graph it was
given, and no graph-level lowering exists, so the compile step refuses and says
why rather than synthesising a stand-in neuron and presenting it as the caller's
network.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from sc_neurocore.studio.synthesis import EdaProcessLimits


#: The route every pipeline result names.
PIPELINE_ROUTE = "graph → simulate → lower → co-simulate → synthesise"

#: Steps the pipeline co-simulates at most. A longer graph is co-simulated over
#: its first steps; the receipt states how many.
PIPELINE_COSIM_STEP_LIMIT = 2000

#: The fixed-point formats a pipeline may compile to, as (width, fraction bits).
PIPELINE_Q_FORMATS: dict[str, tuple[int, int]] = {"Q8.8": (16, 8), "Q16.16": (32, 16)}


def run_pipeline(
    graph: dict[str, Any],
    target: str = "ice40",
    *,
    q_format: str = "Q8.8",
    process_limits: EdaProcessLimits | None = None,
) -> dict[str, Any]:
    """Validate, simulate, lower, co-simulate and synthesise a Studio network.

    The hardware is the network the caller drew: the graph is lowered with each
    catalogue model's own step (:mod:`sc_neurocore.studio.network_hardware`) or
    refused with every reason it cannot be. The compiled RTL is then run beside
    its bit-true model and the Studio's own run
    (:mod:`sc_neurocore.studio.network_hardware_cosim`); synthesis runs only
    when the RTL reproduces its model on every co-simulated step. The result's
    ``trace`` binds the lowering's input digest, the RTL, the model and the
    synthesised source.

    Before the lowering existed the step compiled one hardcoded leaky
    integrate-and-fire equation, ignoring the graph, and later refused every
    graph; neither was the caller's network.

    Parameters
    ----------
    graph:
        Studio network graph payload.
    target:
        Studio synthesis target identifier.
    q_format:
        ``Q8.8`` or ``Q16.16``, the fixed-point format of the hardware.
    process_limits:
        Optional host-supported CPU and address-space ceilings for the
        downstream synthesis child process.

    Returns
    -------
    dict[str, Any]
        ``success``, the ``step`` it ended at, each step's payload under
        ``steps``, and ``trace`` once the RTL was built; a stop carries
        ``error`` and, when the lowering refused, every ``reasons`` entry.

    Raises
    ------
    ValueError
        When ``q_format`` is not one the pipeline compiles to.
    """
    from sc_neurocore.studio.network_graph import simulate_graph, validate_graph
    from sc_neurocore.studio.network_hardware import (
        HardwareLoweringRefused,
        lower_graph,
        lowering_public_dict,
    )
    from sc_neurocore.studio.network_hardware_cosim import (
        HardwareCosimUnavailable,
        compile_lowered,
        cosimulate,
        synthesis_source,
    )
    from sc_neurocore.studio import synthesis as synthesis_module
    from sc_neurocore.studio.synthesis import run_synthesis
    from sc_neurocore.studio.synthesis_capacity import capacity_device_name, capacity_sentence

    if q_format not in PIPELINE_Q_FORMATS:
        raise ValueError(f"q_format must be one of {sorted(PIPELINE_Q_FORMATS)}, got {q_format!r}")
    steps: dict[str, Any] = {}

    errors = validate_graph(graph)
    if errors:
        return {"success": False, "step": "validate", "errors": errors}
    steps["validate"] = {"passed": True}

    sim_result = simulate_graph(graph)
    if not sim_result.get("success"):
        return {"success": False, "step": "simulate", "errors": sim_result.get("errors", [])}
    steps["simulate"] = {
        "n_spikes": sim_result.get("n_spikes", 0),
        "n_total": sim_result.get("n_total", 0),
    }

    def stopped(step: str, error: str, **extra: Any) -> dict[str, Any]:
        return {
            "success": False,
            "step": step,
            "target": target,
            "steps": steps,
            "error": error,
            "pipeline": PIPELINE_ROUTE,
            **extra,
        }

    data_width, fraction = PIPELINE_Q_FORMATS[q_format]
    try:
        lowered = lower_graph(graph, data_width=data_width, fraction=fraction)
    except HardwareLoweringRefused as refused:
        return stopped(
            "lower",
            "the graph cannot be lowered to hardware exactly",
            reasons=list(refused.reasons),
        )
    steps["lower"] = lowering_public_dict(lowered)

    try:
        compiled = compile_lowered(lowered)
    except ValueError as exc:
        return stopped("compile", f"the network compiler refused the lowered graph: {exc}")
    try:
        with tempfile.TemporaryDirectory(prefix="sc_pipeline_") as workdir:
            cosim = cosimulate(
                lowered,
                graph,
                Path(workdir),
                steps=min(lowered.spec.n_steps, PIPELINE_COSIM_STEP_LIMIT),
                compiled=compiled,
            )
    except HardwareCosimUnavailable as exc:
        return stopped("cosimulate", str(exc))
    steps["cosimulate"] = cosim.to_public_dict()
    if not cosim.rtl_matches_model:
        return stopped(
            "cosimulate",
            "the compiled RTL does not reproduce its bit-true model; it is not the lowered network",
        )

    source = synthesis_source(compiled)
    synthesis = run_synthesis(source, target, process_limits=process_limits)
    steps["synthesise"] = synthesis
    trace = {
        "input_sha256": cosim.input_sha256,
        "rtl_sha256": cosim.rtl_sha256,
        "bit_true_model_sha256": cosim.model_sha256,
        "synthesis_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }
    if not synthesis.get("success"):
        # The failure used to leave the result without an error, and the
        # canvas said "Pipeline failed at synthesise: unknown".
        if synthesis.get("timed_out"):
            return stopped(
                "synthesise",
                f"Yosys did not finish within its {synthesis_module.SYNTHESIS_TIMEOUT_SECONDS} s "
                f"limit for this {steps['simulate']['n_total']}-neuron network, so whether it "
                f"fits the {capacity_device_name(target)} is unknown; synthesis time grows "
                "quickly with network size, so try fewer neurons",
                trace=trace,
            )
        return stopped(
            "synthesise",
            f"synthesis produced no netlist: {str(synthesis.get('error') or 'no message')[:300]}",
            trace=trace,
        )
    if synthesis.get("fits_device") is False:
        # A netlist the device cannot hold is not a completed pipeline: a
        # 20-neuron network needed 6237 LUTs of the UP5K's 5280 and was
        # reported "Pipeline complete".
        return stopped(
            "fit",
            "synthesis succeeded, but " + capacity_sentence(target, synthesis["exceeds_capacity"]),
            trace=trace,
        )
    return {
        "success": bool(synthesis.get("success")),
        "step": "synthesise",
        "target": target,
        "steps": steps,
        "trace": trace,
        "pipeline": PIPELINE_ROUTE,
    }
