# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio hardware path and guarantee classification split contract

"""The guarantee classification left the hardware path; every historical import still reaches it.

``sc_neurocore.studio.hardware_path`` held two responsibilities: running one model
through the stages to synthesis, and classifying what the emitted RTL is
guaranteed by. The classification now lives in
``sc_neurocore.studio.model_guarantee``. The hardware path re-exports its public
names as the same objects, so code and pickles that name
``sc_neurocore.studio.hardware_path.model_guarantee_summary`` keep working, and
its guarantee stage records the classification the new module returns.
"""

from __future__ import annotations

import pickle

import sc_neurocore.studio.hardware_path as hardware_path
import sc_neurocore.studio.model_guarantee as model_guarantee

PUBLIC_GUARANTEE_NAMES = (
    "MODEL_GUARANTEE_SCHEMA_VERSION",
    "formal_guarantee",
    "model_guarantee_summary",
    "readiness_labels",
    "render_model_guarantee_text",
)


def test_historical_hardware_path_imports_are_the_guarantee_module_objects() -> None:
    """Each public guarantee name of the hardware path is the guarantee module's own object."""
    for name in PUBLIC_GUARANTEE_NAMES:
        assert getattr(hardware_path, name) is getattr(model_guarantee, name), name
    assert set(PUBLIC_GUARANTEE_NAMES) == set(model_guarantee.__all__)
    assert set(PUBLIC_GUARANTEE_NAMES) <= set(hardware_path.__all__)


def test_the_classification_is_defined_in_its_own_module() -> None:
    """The classifying functions are owned by the guarantee module, by name and by pickle."""
    for name in PUBLIC_GUARANTEE_NAMES[1:]:
        function = getattr(model_guarantee, name)
        assert function.__module__ == "sc_neurocore.studio.model_guarantee", name
        assert function.__qualname__ == name
        assert pickle.loads(pickle.dumps(getattr(hardware_path, name))) is function


def test_the_hardware_path_no_longer_carries_the_classification_helpers() -> None:
    """The hardware path keeps neither the formal comparison table nor the helpers it moved."""
    names = vars(hardware_path)
    for name in ("_FORMAL_COMPARED", "_pairs", "_sha256", "GUARANTEE_CLASSES", "sealed_detail"):
        assert name not in names, name
    assert vars(model_guarantee)["_FORMAL_COMPARED"] == (
        ("module", "module"),
        ("profile", "schema"),
        ("q_format", "q_format"),
    )
