# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A cited, runnable notebook generated from a replay pack

"""Turn a sealed replay pack into a Jupyter notebook another researcher can run.

The notebook carries the pack inline — no path on the author's machine is
needed — and cites the catalogue model the experiment ran, from the model's
own descriptor. Running it replays the pack through the public replay entry
point and prints the verdict against the sealed expectation. A replay on other
software versions is admitted and its runtime differences are printed, rather
than refused, because a second researcher's installation is expected to
differ; the verdict still compares every spike and state sample.

The notebook runs the software model only. It says so in its first cell: no
fixed-point, RTL, synthesis or board step is part of it, and it establishes
nothing about hardware.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

NOTEBOOK_FORMAT = 4
NOTEBOOK_FORMAT_MINOR = 5


def model_citation(class_name: str) -> str:
    """Cite one catalogue model from its own descriptor, or say it names no source.

    Parameters
    ----------
    class_name:
        The catalogue class name.

    Returns
    -------
    str
        A Markdown sentence naming the model and its source.
    """
    from sc_neurocore.neurons.model_catalogue import load_descriptor

    descriptor = load_descriptor(str(class_name))
    if descriptor is None or not descriptor.provenance.is_citeable:
        return f"`{class_name}` from the SC-NeuroCore catalogue; its descriptor names no citeable source."
    # A citeable provenance has authors, a year and a DOI by definition.
    provenance = descriptor.provenance
    parts = [", ".join(provenance.authors), f"({provenance.year})"]
    if provenance.paper_title:
        parts.append(f"*{provenance.paper_title}*")
    parts.append(f"doi:{provenance.doi}")
    return f"`{class_name}` follows " + " ".join(parts) + "."


def _citation(pack: Mapping[str, Any]) -> str:
    model = pack.get("experiment", {}).get("model") or {}
    class_name = model.get("class_name")
    if not class_name:
        return "Custom equations written in the Studio's equation playground; no published source is attached."
    return model_citation(str(class_name))


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


def notebook_from_pack(pack: Mapping[str, Any]) -> dict[str, Any]:
    """Return a Jupyter notebook (nbformat 4) that cites and replays ``pack``.

    Parameters
    ----------
    pack:
        A sealed replay pack, as :func:`~sc_neurocore.studio.replay_pack.build_replay_pack`
        returns it.

    Returns
    -------
    dict
        The notebook document, ready to be written as ``.ipynb`` JSON.
    """
    expectation = pack.get("expectation", {})
    intro = "\n".join(
        [
            "# Replaying a sealed SC-NeuroCore experiment",
            "",
            f"**Source.** {_citation(pack)}",
            "",
            f"**Experiment.** `{pack.get('experiment_sha256', '')}` "
            f"(scientific identity `{pack.get('experiment_identity_sha256', '')}`).",
            "",
            "**What this notebook does.** It carries the sealed pack inline, replays it through "
            "`sc_neurocore.studio.replay_pack.replay_pack`, and compares every spike and state "
            "sample with the sealed expectation.",
            "",
            "**What it does not do.** It runs the software model only. No fixed-point, RTL, "
            "synthesis or board step is part of it, and it establishes nothing about hardware.",
        ]
    )
    load = "\n".join(
        [
            "import json",
            "",
            "from sc_neurocore.studio.replay_pack import replay_pack",
            "",
            "# The sealed pack, inline: nothing here depends on a file on the author's machine.",
            f"PACK = json.loads({json.dumps(json.dumps(pack, sort_keys=True))})",
            'print(PACK["source"], PACK["experiment_sha256"])',
        ]
    )
    replay = "\n".join(
        [
            "# Another installation may run other software versions; they are reported, not refused.",
            "report = replay_pack(PACK, allow_runtime_drift=True)",
            'print("verdict:", report["verdict"])',
            'print("worst state deviation:", report["worst_state_deviation"])',
            'print("runtime differences:", report["runtime_differences"] or "none")',
            'for difference in report["differences"]:',
            '    print("difference:", difference)',
        ]
    )
    summary = (
        f"The sealed run produced {expectation.get('spike_count', 0)} spikes in "
        f"{expectation.get('n_steps', 0)} steps. "
        "A verdict of `match` or `match-within-tolerance` reproduces it; `mismatch` lists what differs."
    )
    return {
        "nbformat": NOTEBOOK_FORMAT,
        "nbformat_minor": NOTEBOOK_FORMAT_MINOR,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "sc_neurocore": {
                "replay_pack_schema": pack.get("schema_version"),
                "experiment_sha256": pack.get("experiment_sha256"),
            },
        },
        "cells": [_markdown(intro), _code(load), _code(replay), _markdown(summary)],
    }


__all__ = ["NOTEBOOK_FORMAT", "model_citation", "notebook_from_pack"]
