# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public model discovery contracts

"""Exercise immutable discovery and actual lazy model imports."""

from __future__ import annotations

import importlib
import operator
import subprocess
import sys
from collections.abc import MutableMapping
from typing import cast

import pytest

from sc_neurocore.neurons import models
from sc_neurocore.neurons.model_registry import MODEL_MODULES


def test_registry_resolves_every_public_model_class() -> None:
    """Resolve every catalogue identity through both supported import paths."""
    assert set(MODEL_MODULES) == set(models.__all__)
    for name, module_name in MODEL_MODULES.items():
        module = importlib.import_module(f"sc_neurocore.neurons.models.{module_name}")
        cls = getattr(module, name)
        assert isinstance(cls, type)
        assert getattr(models, name) is cls


def test_registry_cannot_be_changed_by_a_consumer() -> None:
    """Refuse replacement, insertion and deletion without changing discovery."""
    before = tuple(MODEL_MODULES.items())
    attempted_mutation = cast(MutableMapping[str, str], MODEL_MODULES)
    with pytest.raises(TypeError):
        operator.setitem(attempted_mutation, "AdExNeuron", "poisson")
    with pytest.raises(TypeError):
        operator.setitem(attempted_mutation, "ConsumerNeuron", "adex")
    with pytest.raises(TypeError):
        operator.delitem(attempted_mutation, "AdExNeuron")
    assert tuple(MODEL_MODULES.items()) == before


def test_historical_import_alias_does_not_become_a_catalogue_identity() -> None:
    """Keep the historical import available without inflating discovery."""
    assert isinstance(models.KilincBhattMapNeuron, type)
    assert "KilincBhattMapNeuron" not in MODEL_MODULES
    with pytest.raises(KeyError, match="KilincBhattMapNeuron"):
        operator.getitem(MODEL_MODULES, "KilincBhattMapNeuron")
    with pytest.raises(KeyError, match="UnregisteredNeuron"):
        operator.getitem(MODEL_MODULES, "UnregisteredNeuron")


def test_discovery_in_a_fresh_interpreter_keeps_model_imports_lazy() -> None:
    """Read the registry before any model implementation or training backend loads."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from sc_neurocore.neurons.model_registry import MODEL_MODULES; "
            "assert MODEL_MODULES['AdExNeuron'] == 'adex'; "
            "assert not any(n.startswith('sc_neurocore.neurons.models.') for n in sys.modules); "
            "assert 'torch' not in sys.modules; "
            "print('lazy-model-discovery-ok')",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert completed.stdout.strip() == "lazy-model-discovery-ok"
