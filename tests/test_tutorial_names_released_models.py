# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_AUTOFIT_TUTORIAL = REPO_ROOT / "docs" / "tutorials" / "47_autofit.md"


def _documented_fittable_models() -> list[str]:
    """Return the model identities the autofit tutorial's table lists."""
    text = _AUTOFIT_TUTORIAL.read_text(encoding="utf-8")
    table = text.split("## Fittable Models", 1)[1].split("\n##", 1)[0]
    return [
        name
        for name in re.findall(r"^\|\s*([A-Za-z0-9]+)\s*\|", table, re.MULTILINE)
        if name != "Model"
    ]


def test_autofit_tutorial_lists_exactly_the_models_the_sweep_tries() -> None:
    """The documented sweep and the shipped sweep must be the same set.

    The table named `LIFNeuron` and `IzhikevichNeuron`, neither of which exists,
    while the code's own list carried two different phantoms — so a reader was
    told about models the sweep never tried, and the sweep skipped models the
    reader was never told about. Both lists are now bound to the same
    identities, and this test is what keeps them so.

    Deliberately narrow. A general rule that every capitalised name in the
    tutorials must resolve is not true: the surrogate-gradient tutorial names
    third-party `torch` classes, the NIR tutorial names `norse` primitives, and
    the custom-model tutorial defines its own teaching classes. Those are all
    correct, and a wider rule would call them defects.
    """
    from sc_neurocore.autofit.fitter import _FITTABLE_MODELS

    # The set, not the order: the table is arranged by increasing complexity for
    # a reader, the list by the order the sweep grew. Pinning the order would
    # couple a presentational choice to an implementation one.
    documented = _documented_fittable_models()
    assert len(documented) == len(set(documented)), "the table repeats a model"
    assert set(documented) == set(_FITTABLE_MODELS)


def test_every_documented_fittable_model_is_a_catalogue_identity() -> None:
    """Each name in the table resolves in the released catalogue."""
    from sc_neurocore.neurons.models import _CLASS_TO_MODULE

    unknown = sorted(name for name in _documented_fittable_models() if name not in _CLASS_TO_MODULE)

    assert unknown == [], f"tutorial names models absent from the catalogue: {unknown}"
