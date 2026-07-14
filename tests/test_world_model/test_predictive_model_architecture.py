# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Predictive-model responsibility and compatibility contracts

"""Facade identity, API, ownership, import-DAG, and structured-solve contracts."""

from __future__ import annotations

import ast
import inspect
import pickle
from collections.abc import Callable
from pathlib import Path

import sc_neurocore.world_model as world_model_package
import sc_neurocore.world_model.predictive_model as predictive_model
import pytest
from sc_neurocore.world_model import _lgssm_backends as backend_runtime


WORLD_MODEL_ROOT = Path(predictive_model.__file__).resolve().parent
MODULE_PATHS = {
    "predictive_model": WORLD_MODEL_ROOT / "predictive_model.py",
    "_lgssm_types": WORLD_MODEL_ROOT / "_lgssm_types.py",
    "_lgssm_backends": WORLD_MODEL_ROOT / "_lgssm_backends.py",
    "_lgssm_filter": WORLD_MODEL_ROOT / "_lgssm_filter.py",
    "_lgssm_smoothing": WORLD_MODEL_ROOT / "_lgssm_smoothing.py",
    "_lgssm_em": WORLD_MODEL_ROOT / "_lgssm_em.py",
    "_predictive_world_model": WORLD_MODEL_ROOT / "_predictive_world_model.py",
}
PUBLIC_CLASSES = (
    predictive_model.LinearGaussianSSM,
    predictive_model.FilterResult,
    predictive_model.KalmanFilter,
    predictive_model.SmoothResult,
    predictive_model.RTSSmoother,
    predictive_model.EMLearner,
    predictive_model.PredictiveWorldModel,
)


def _parameter_names(callable_object: Callable[..., object]) -> tuple[str, ...]:
    return tuple(inspect.signature(callable_object).parameters)


def _local_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module is not None
        and node.module in MODULE_PATHS
    }


def test_facade_exports_exact_historical_public_class_surface() -> None:
    assert predictive_model.__all__ == [
        "LinearGaussianSSM",
        "FilterResult",
        "KalmanFilter",
        "SmoothResult",
        "RTSSmoother",
        "EMLearner",
        "PredictiveWorldModel",
    ]
    assert world_model_package.PredictiveWorldModel is predictive_model.PredictiveWorldModel


def test_public_classes_preserve_historical_pickle_identity() -> None:
    for public_class in PUBLIC_CLASSES:
        assert public_class.__module__ == "sc_neurocore.world_model.predictive_model"
        assert pickle.loads(pickle.dumps(public_class)) is public_class


def test_facade_reads_live_historical_rust_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend_runtime, "_HAS_RUST_LGSSM", False)
    assert predictive_model._HAS_RUST_LGSSM is False
    monkeypatch.setattr(backend_runtime, "_HAS_RUST_LGSSM", True)
    assert predictive_model._HAS_RUST_LGSSM is True
    with pytest.raises(AttributeError, match="has no attribute"):
        predictive_model._unknown_legacy_attribute


def test_public_constructor_and_method_parameter_names_remain_compatible() -> None:
    assert _parameter_names(predictive_model.LinearGaussianSSM) == (
        "A",
        "B",
        "C",
        "D",
        "Q",
        "R",
        "mu_0",
        "Sigma_0",
    )
    assert _parameter_names(predictive_model.FilterResult) == (
        "means",
        "covariances",
        "pred_means",
        "pred_covariances",
        "log_likelihood",
    )
    assert _parameter_names(predictive_model.KalmanFilter.filter) == (
        "self",
        "observations",
        "controls",
        "backend",
    )
    assert _parameter_names(predictive_model.RTSSmoother.smooth) == (
        "self",
        "filter_result",
    )
    assert _parameter_names(predictive_model.EMLearner.fit) == (
        "self",
        "observations",
        "initial_model",
        "controls",
        "backend",
    )
    assert _parameter_names(predictive_model.PredictiveWorldModel) == (
        "state_dim",
        "action_dim",
        "seed",
    )


def test_each_public_class_has_one_responsibility_owner() -> None:
    expected = {
        "predictive_model": set(),
        "_lgssm_types": {"LinearGaussianSSM", "FilterResult", "SmoothResult"},
        "_lgssm_backends": set(),
        "_lgssm_filter": {"KalmanFilter"},
        "_lgssm_smoothing": {"RTSSmoother"},
        "_lgssm_em": {"EMLearner"},
        "_predictive_world_model": {"PredictiveWorldModel"},
    }
    for module_name, path in MODULE_PATHS.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        public_definitions = {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
        }
        assert public_definitions == expected[module_name]


def test_private_module_import_graph_is_acyclic_and_one_way() -> None:
    expected_imports = {
        "predictive_model": {
            "_lgssm_backends",
            "_lgssm_em",
            "_lgssm_filter",
            "_lgssm_smoothing",
            "_lgssm_types",
            "_predictive_world_model",
        },
        "_lgssm_types": set(),
        "_lgssm_backends": {"_lgssm_types"},
        "_lgssm_filter": {"_lgssm_backends", "_lgssm_types"},
        "_lgssm_smoothing": {"_lgssm_types"},
        "_lgssm_em": {"_lgssm_filter", "_lgssm_smoothing", "_lgssm_types"},
        "_predictive_world_model": {"_lgssm_types"},
    }
    assert {name: _local_imports(path) for name, path in MODULE_PATHS.items()} == expected_imports


def test_numerical_modules_contain_no_explicit_matrix_inverse() -> None:
    for module_name in ("_lgssm_filter", "_lgssm_smoothing", "_lgssm_em"):
        source = MODULE_PATHS[module_name].read_text(encoding="utf-8")
        assert "np.linalg.inv" not in source
        assert ".inverse(" not in source


def test_responsibility_modules_remain_bounded() -> None:
    ceilings = {
        "predictive_model": 90,
        "_lgssm_types": 450,
        "_lgssm_backends": 510,
        "_lgssm_filter": 190,
        "_lgssm_smoothing": 140,
        "_lgssm_em": 300,
        "_predictive_world_model": 230,
    }
    for module_name, path in MODULE_PATHS.items():
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        assert line_count <= ceilings[module_name], (module_name, line_count)
