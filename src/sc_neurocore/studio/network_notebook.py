# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A tutorial notebook that rebuilds a Studio network step by step

"""Turn a network drawn in the Studio into a notebook that builds it by hand.

The Studio lowers a graph to the public ``sc_neurocore.network`` objects a
user script would build. This module writes that script out, one population
and one projection at a time, as a notebook a learner can read and run: the
constructor keywords, the connectivity rule and seed, the delay in whole
steps, the stimulus and the order the objects join the network are all
visible. The last cell compares the spikes and the connectivity of the run
with the Studio's own run, sealed into the notebook as digests, so the
tutorial shows whether it reproduced the network rather than asserting it.

The notebook runs the software network only. It says so, and it names the
Rust network runner the Studio rejects for graphs, and why.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from sc_neurocore.studio.network_execution import simulate_graph_spec
from sc_neurocore.studio.network_graph_spec import GraphSpec, PopulationSpec, ProjectionSpec
from sc_neurocore.studio.replay_notebook import (
    NOTEBOOK_FORMAT,
    NOTEBOOK_FORMAT_MINOR,
    model_citation,
)

NETWORK_NOTEBOOK_KIND = "sc-neurocore.network-tutorial.v1"


def spike_events_sha256(result: Mapping[str, Any]) -> str:
    """Digest every spike event of a graph result, in network-wide neuron indices.

    Parameters
    ----------
    result:
        A ``studio.network-graph-result.v1`` payload.

    Returns
    -------
    str
        SHA-256 over the int64 bytes of the event steps, then the event
        neurons, sorted by step and then neuron.
    """
    steps: list[np.ndarray[Any, Any]] = []
    neurons: list[np.ndarray[Any, Any]] = []
    for population in result["populations"]:
        steps.append(np.asarray(population["events"]["step"], dtype=np.int64))
        neurons.append(
            np.asarray(population["events"]["neuron"], dtype=np.int64) + int(population["offset"])
        )
    return _events_digest(np.concatenate(steps), np.concatenate(neurons))


def _events_digest(steps: np.ndarray[Any, Any], neurons: np.ndarray[Any, Any]) -> str:
    order = np.lexsort((neurons, steps))
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(steps[order], dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(neurons[order], dtype=np.int64).tobytes())
    return digest.hexdigest()


def _literal(value: object) -> str:
    """Python source for a constructor value, exact for floats."""
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, int):
        return repr(int(value))
    if isinstance(value, float):
        return repr(float(value))
    raise TypeError(f"no literal for {type(value).__name__}")


def _params(population: PopulationSpec) -> str:
    items = sorted(population.inputs.constructor_kwargs.items())
    return "{" + ", ".join(f"{name!r}: {_literal(value)}" for name, value in items) + "}"


def _population_code(population: PopulationSpec) -> str:
    pid = repr(population.id)
    cls = population.inputs.cls.__name__
    lines = [
        f"# {population.label}: {population.count} {cls} neurons, {population.neuron_type}.",
        f"populations[{pid}] = Population(",
        f"    {cls}, {population.count}, params={_params(population)}, label={pid}",
        ")",
        f"monitors[{pid}] = SpikeMonitor(populations[{pid}], label={'spikes_' + population.id!r})",
    ]
    drive = population.drive
    if drive.kind == "constant":
        lines += [
            f"# Constant drive: {drive.current!r} into every neuron at every step.",
            f"stimulus = StepCurrent(0, N_STEPS, {_literal(drive.current)})",
            f"stimulus.target = populations[{pid}]",
            "stimuli.append(stimulus)",
        ]
    elif drive.kind == "poisson":
        lines += [
            f"# Poisson drive: each neuron receives {drive.weight!r} when its own "
            f"{drive.rate_hz!r} Hz process fires.",
            f"stimulus = PoissonInput({population.count}, {_literal(drive.rate_hz)}, "
            f"{_literal(drive.weight)}, dt=DT_S, seed={drive.seed})",
            f"stimulus.target = populations[{pid}]",
            "stimuli.append(stimulus)",
        ]
    else:
        lines.append("# No external drive: this population is driven only by projections.")
    return "\n".join(lines)


def _projection_code(projection: ProjectionSpec) -> str:
    source, target = repr(projection.source), repr(projection.target)
    if projection.rule == "all_to_all":
        rule = f"all_to_all(populations[{source}].n, populations[{target}].n, {_literal(projection.weight)})"
        described = "every source neuron to every target neuron"
    else:
        rule = (
            f"random_connectivity(populations[{source}].n, populations[{target}].n, "
            f"{_literal(projection.probability)}, {_literal(projection.weight)}, {projection.seed})"
        )
        described = f"each pair with probability {projection.probability!r}, seed {projection.seed}"
    lines = [
        f"# {projection.id}: {projection.source} -> {projection.target}, {described}.",
        f"# Weight {projection.weight!r} ({projection.sign}); delay {projection.delay_ms!r} ms "
        f"= {projection.delay_steps} whole steps.",
        f"indptr, indices, data = {rule}",
    ]
    if projection.source == projection.target and not projection.autapses:
        lines.append("indptr, indices, data = without_autapses(indptr, indices, data)")
    lines += [
        f"projections[{projection.id!r}] = Projection(",
        f"    populations[{source}],",
        f"    populations[{target}],",
        f"    weight={_literal(projection.weight)},",
        f"    probability={_literal(projection.probability)},",
        f"    delay={_literal(float(projection.delay_steps))},",
        "    topology=(indptr, indices, data),",
        f"    seed={projection.seed},",
        ")",
    ]
    return "\n".join(lines)


_WITHOUT_AUTAPSES = '''

def without_autapses(indptr, indices, data):
    """Drop the diagonal of a self-projection, as the Studio does unless autapses are declared."""
    rows = np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))
    keep = rows != indices
    if keep.all():
        return indptr, indices, data
    new_indptr = np.zeros(len(indptr), dtype=np.int64)
    np.cumsum(np.bincount(rows[keep], minlength=len(indptr) - 1), out=new_indptr[1:])
    return new_indptr, indices[keep], data[keep]'''


def _markdown(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def _code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


def _setup_code(spec: GraphSpec) -> str:
    classes = sorted({(p.inputs.cls.__module__, p.inputs.cls.__name__) for p in spec.populations})
    imports = [
        "import hashlib",
        "",
        "import numpy as np",
        "",
        "from sc_neurocore.network import (",
        "    Network,",
        "    PoissonInput,",
        "    Population,",
        "    Projection,",
        "    SpikeMonitor,",
        "    StepCurrent,",
        ")",
        "from sc_neurocore.network.topology import all_to_all, random_connectivity",
    ]
    imports += [f"from {module} import {name}" for module, name in classes]
    body = "\n".join(imports) + "\n\n"
    body += (
        f"DT_MS = {_literal(spec.dt)}  # one step, in ms\n"
        f"N_STEPS = {spec.n_steps}  # {spec.n_steps * spec.dt:g} ms\n"
        "DT_S = DT_MS / 1000.0  # the network runs in seconds\n\n"
        "populations = {}\nmonitors = {}\nprojections = {}\nstimuli = []"
    )
    if any(p.source == p.target and not p.autapses for p in spec.projections):
        body += _WITHOUT_AUTAPSES
    return body


def _run_code(spec: GraphSpec) -> str:
    return "\n".join(
        [
            "# The objects join the network in the Studio's order: populations,",
            "# projections, monitors, stimuli. The reference Python loop runs it.",
            f"network = Network(seed={spec.seed})",
            "for population in populations.values():",
            "    network.add(population)",
            "for projection in projections.values():",
            "    network.add(projection)",
            "for monitor in monitors.values():",
            "    network.add(monitor)",
            "for stimulus in stimuli:",
            "    network.add(stimulus)",
            'network.run(duration=N_STEPS * DT_S, dt=DT_S, backend="python")',
        ]
    )


def _check_code(spec: GraphSpec, sealed: Mapping[str, Any]) -> str:
    offsets = {p.id: p.offset for p in spec.populations}
    return "\n".join(
        [
            f"SEALED = json.loads({json.dumps(json.dumps(sealed, sort_keys=True))})",
            f"OFFSETS = {offsets!r}  # first network-wide index of each population",
            "",
            "steps, neurons = [], []",
            "for population_id, monitor in monitors.items():",
            "    population_steps, population_neurons = monitor.raster_data()",
            "    steps.append(np.asarray(population_steps, dtype=np.int64))",
            "    neurons.append(np.asarray(population_neurons, dtype=np.int64) + OFFSETS[population_id])",
            "steps, neurons = np.concatenate(steps), np.concatenate(neurons)",
            "order = np.lexsort((neurons, steps))",
            "digest = hashlib.sha256()",
            "digest.update(steps[order].tobytes())",
            "digest.update(neurons[order].tobytes())",
            'print("spikes:", steps.size, "sealed:", SEALED["n_spikes"])',
            'print("spikes match:", digest.hexdigest() == SEALED["spikes_sha256"])',
            "",
            "for projection_id, projection in projections.items():",
            "    csr = hashlib.sha256()",
            "    csr.update(np.ascontiguousarray(projection.indptr, dtype=np.int64).tobytes())",
            "    csr.update(np.ascontiguousarray(projection.indices, dtype=np.int64).tobytes())",
            "    csr.update(np.ascontiguousarray(projection.data, dtype=np.float64).tobytes())",
            '    same = csr.hexdigest() == SEALED["csr_sha256"][projection_id]',
            '    print(f"{projection_id}: {projection.n_synapses} synapses, connectivity match: {same}")',
        ]
    )


def network_notebook(spec: GraphSpec) -> dict[str, Any]:
    """Run a resolved graph and return a tutorial notebook that rebuilds it.

    Parameters
    ----------
    spec:
        A graph resolved by :func:`~sc_neurocore.studio.network_graph_spec.resolve_graph`.

    Returns
    -------
    dict
        An nbformat 4 notebook. Its metadata carries the graph digest and the
        sealed spike and connectivity digests of the Studio run.

    Raises
    ------
    GraphExecutionFailure
        When the graph fails while running.
    """
    result = simulate_graph_spec(spec)
    public = result["spec"]
    sealed = {
        "graph_sha256": public["graph_sha256"],
        "n_spikes": result["n_spikes"],
        "spikes_sha256": spike_events_sha256(result),
        "csr_sha256": {
            item["id"]: item["csr_sha256"] for item in result["topology"]["projections"]
        },
    }
    models = sorted({p.model for p in spec.populations})
    rejected = result["execution"]["backend"]["rejected"][0]
    intro = "\n".join(
        [
            "# Building a Studio network by hand",
            "",
            "This notebook rebuilds, with the public `sc_neurocore.network` API, the network "
            f"`{public['graph_sha256'][:12]}` drawn in the SC-NeuroCore Studio: "
            f"{len(spec.populations)} populations, {spec.n_neurons} neurons, "
            f"{len(spec.projections)} projections, {spec.n_steps} steps of {spec.dt:g} ms.",
            "",
            "**Sources.**",
            "",
            *[f"- {model_citation(model)}" for model in models],
            "",
            "**What it shows.** Each population and projection is one visible constructor "
            "call; the last cell compares the spikes and the connectivity with the Studio's "
            "own run of the same graph.",
            "",
            "**What it does not do.** It runs the software network only: no fixed-point, RTL, "
            "synthesis or board step is part of it, and it establishes nothing about "
            f"hardware. The `{rejected['name']}` is not used because it {rejected['reason']}.",
        ]
    )
    cells = [
        _markdown(intro),
        _code(_setup_code(spec)),
        _markdown(
            "## Populations\n\nEach population is a catalogue model with the constructor "
            "keywords the Studio validated, one spike monitor, and its external drive."
        ),
        *[_code(_population_code(population)) for population in spec.populations],
    ]
    if spec.projections:
        cells.append(
            _markdown(
                "## Projections\n\nA spike at step *t* reaches its targets at step "
                "*t* + 1 + delay: one step of inherent latency, then the delay in whole "
                "steps. Each source spike drives its targets with the weight for one step."
            )
        )
        cells += [_code(_projection_code(projection)) for projection in spec.projections]
    cells += [
        _markdown("## Run"),
        _code(_run_code(spec)),
        _markdown(
            "## Check against the Studio run\n\n"
            f"The Studio's run produced {result['n_spikes']} spikes. `spikes match: True` "
            "means every spike event — step and neuron — is the same."
        ),
        _code("import json\n\n" + _check_code(spec, sealed)),
    ]
    return {
        "nbformat": NOTEBOOK_FORMAT,
        "nbformat_minor": NOTEBOOK_FORMAT_MINOR,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "sc_neurocore": {"kind": NETWORK_NOTEBOOK_KIND, **sealed},
        },
        "cells": cells,
    }


__all__ = ["NETWORK_NOTEBOOK_KIND", "network_notebook", "spike_events_sha256"]
