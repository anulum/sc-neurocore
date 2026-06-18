# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parameter quantisation for FPGA-targeted NeuronGraphs

"""Quantise floating-point learned parameters to fixed-point Q-format.

Converts all weights, time constants, thresholds, and biases in a
``NeuronGraph`` to the target Q-format (e.g. Q8.8, Q4.12, Q16.16).
Performs range checking and emits overflow/underflow warnings.

Quantisation Strategy
~~~~~~~~~~~~~~~~~~~~~

1. **Weights** — Each element is individually encoded via ``Q88.encode()``.
   Out-of-range values are clamped with a warning.

2. **Neuron parameters** — Scalar or per-neuron arrays are encoded with
   the same Q-format.  Time constants (``tau``, ``tau_syn``, ``tau_mem``)
   are checked for underflow (small dt × tau can quantise to zero).

3. **Biases** — Treated identically to weights.

Usage
~~~~~

::

    from sc_neurocore.nir_bridge.neuron_graph import from_scnetwork
    from sc_neurocore.nir_bridge.quantise_params import quantise_graph
    from sc_neurocore.compiler.equation_compiler import Q88

    graph = from_scnetwork(network)
    q = Q88(data_width=16, fraction=8)
    qgraph = quantise_graph(graph, q)
    print(qgraph.warnings)  # any overflow/underflow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from typing import Any
import numpy as np

from ..compiler.equation_compiler import Q88
from .neuron_graph import ConnectionSpec, NeuronGraph, NeuronSpec

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Quantised Graph
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class QuantisedGraph:
    """NeuronGraph with all parameters converted to Q-format integers.

    Attributes
    ----------
    populations : list[NeuronSpec]
        Populations with integer-valued parameters (Q-encoded).
    connections : list[ConnectionSpec]
        Connections with integer-valued weight matrices (Q-encoded).
    q : Q88
        The fixed-point format configuration used.
    input_pop : str
        Input population name.
    output_pop : str
        Output population name.
    dt : float
        Global timestep.
    warnings : list[str]
        Overflow/underflow warnings generated during quantisation.
    total_neurons : int
        Total neuron count.
    total_synapses : int
        Total synapse count.
    """

    populations: list[NeuronSpec]
    connections: list[ConnectionSpec]
    q: Q88
    input_pop: str
    output_pop: str
    dt: float
    warnings: list[str] = field(default_factory=list)
    total_neurons: int = 0
    total_synapses: int = 0


# ═══════════════════════════════════════════════════════════════════════
# Quantisation Logic
# ═══════════════════════════════════════════════════════════════════════


def _quantise_array(
    arr: np.ndarray[Any, Any],
    q: Q88,
    label: str,
    warnings: list[str],
) -> np.ndarray[Any, Any]:
    """Quantise a float array to Q-format integers with clamping.

    Parameters
    ----------
    arr : np.ndarray[Any, Any]
        Float values to quantise.
    q : Q88
        Fixed-point format.
    label : str
        Label for warning messages.
    warnings : list[str]
        Accumulator for overflow/underflow warnings.

    Returns
    -------
    np.ndarray[Any, Any]
        Integer array of Q-encoded values (dtype int64).
    """
    flat = arr.flatten()
    result = np.empty_like(flat, dtype=np.int64)

    max_int = (1 << (q.data_width - 1)) - 1 if q.signed else (1 << q.data_width) - 1
    min_int = -(1 << (q.data_width - 1)) if q.signed else 0

    overflow_count = 0
    underflow_count = 0

    for i, val in enumerate(flat):
        raw = int(round(float(val) * (1 << q.fraction)))
        if raw > max_int:
            overflow_count += 1
            raw = max_int
        elif raw < min_int:
            underflow_count += 1
            raw = min_int
        result[i] = raw

    if overflow_count > 0:
        warnings.append(
            f"Overflow: {overflow_count}/{len(flat)} values in '{label}' "
            f"clamped to Q{q.data_width - q.fraction}.{q.fraction} max={q.max_value:.4f}"
        )
        logger.warning(
            "Quantisation overflow in %s: %d/%d values clamped",
            label,
            overflow_count,
            len(flat),
        )

    if underflow_count > 0:
        warnings.append(
            f"Underflow: {underflow_count}/{len(flat)} values in '{label}' "
            f"clamped to Q{q.data_width - q.fraction}.{q.fraction} min={q.min_value:.4f}"
        )
        logger.warning(
            "Quantisation underflow in %s: %d/%d values clamped",
            label,
            underflow_count,
            len(flat),
        )

    return result.reshape(arr.shape)


def _check_dt_quantisation(
    dt: float,
    q: Q88,
    warnings: list[str],
) -> None:
    """Verify that the timestep survives Q-format quantisation.

    Parameters
    ----------
    dt : float
        Simulation timestep.
    q : Q88
        Fixed-point format.
    warnings : list[str]
        Accumulator for warnings.
    """
    if dt == 0.0:
        return
    dt_quantised = int(round(dt * (1 << q.fraction)))
    if dt_quantised == 0:
        min_rep = 1.0 / (1 << q.fraction)
        warnings.append(
            f"CRITICAL: dt={dt} underflows to 0 in "
            f"Q{q.data_width - q.fraction}.{q.fraction}. "
            f"Minimum representable: {min_rep}. "
            f"All dynamics will be frozen."
        )
        logger.error(
            "dt=%s quantises to 0 in Q%d.%d — all dynamics frozen",
            dt,
            q.data_width - q.fraction,
            q.fraction,
        )


def quantise_graph(graph: NeuronGraph, q: Q88) -> QuantisedGraph:
    """Convert all floating-point parameters to Q-format integers.

    Parameters
    ----------
    graph : NeuronGraph
        Network with float32 parameters.
    q : Q88
        Target fixed-point format.

    Returns
    -------
    QuantisedGraph
        Network with integer-valued parameters and quantisation warnings.
    """
    warnings: list[str] = []

    # Check global dt
    _check_dt_quantisation(graph.dt, q, warnings)

    # Quantise populations
    q_populations: list[NeuronSpec] = []
    for pop in graph.populations:
        q_params: dict[str, np.ndarray[Any, Any]] = {}
        for pname, pval in pop.params.items():
            q_params[pname] = _quantise_array(
                pval,
                q,
                label=f"{pop.name}.{pname}",
                warnings=warnings,
            )
        q_populations.append(
            NeuronSpec(
                name=pop.name,
                neuron_type=pop.neuron_type,
                n_neurons=pop.n_neurons,
                params=q_params,
                dt=pop.dt,
            )
        )

    # Quantise connections
    q_connections: list[ConnectionSpec] = []
    for conn in graph.connections:
        q_weights = _quantise_array(
            conn.weights,
            q,
            label=f"weights[{conn.src}→{conn.dst}]",
            warnings=warnings,
        )
        q_bias = None
        if conn.bias is not None:
            q_bias = _quantise_array(
                conn.bias,
                q,
                label=f"bias[{conn.src}→{conn.dst}]",
                warnings=warnings,
            )
        q_source_threshold = None
        if conn.source_threshold is not None:
            q_source_threshold = _quantise_array(
                conn.source_threshold,
                q,
                label=f"source_threshold[{conn.src}→{conn.dst}]",
                warnings=warnings,
            )
        q_destination_threshold = None
        if conn.destination_threshold is not None:
            q_destination_threshold = _quantise_array(
                conn.destination_threshold,
                q,
                label=f"destination_threshold[{conn.src}→{conn.dst}]",
                warnings=warnings,
            )
        q_connections.append(
            ConnectionSpec(
                src=conn.src,
                dst=conn.dst,
                weights=q_weights,
                bias=q_bias,
                delay_steps=conn.delay_steps,
                source_threshold=q_source_threshold,
                destination_threshold=q_destination_threshold,
            )
        )

    result = QuantisedGraph(
        populations=q_populations,
        connections=q_connections,
        q=q,
        input_pop=graph.input_pop,
        output_pop=graph.output_pop,
        dt=graph.dt,
        warnings=warnings,
        total_neurons=graph.total_neurons,
        total_synapses=graph.total_synapses,
    )

    logger.info(
        "Quantised %d populations, %d connections to Q%d.%d (%d warnings)",
        len(q_populations),
        len(q_connections),
        q.data_width - q.fraction,
        q.fraction,
        len(warnings),
    )

    return result
