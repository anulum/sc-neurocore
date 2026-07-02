# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pretrained model-zoo loader contract tests

"""Contract tests for pretrained model-zoo weight loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sc_neurocore.model_zoo import pretrained as _pt
from sc_neurocore.model_zoo.configs import mnist_classifier
from sc_neurocore.network import Network


def _write_mnist_archive(directory: Path, filename: str) -> None:
    """Write a valid MNIST pretrained-weight archive for loader tests."""
    np.savez(
        directory / filename,
        W0=np.full((784, 128), 7.0, dtype=np.float32),
        W1=np.full((128, 10), -3.0, dtype=np.float32),
    )


def test_load_pretrained_missing_archive_names_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing weight archives report the requested pretrained model name."""
    monkeypatch.setattr(_pt, "_WEIGHTS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="pretrained model 'mnist'"):
        _pt.load_pretrained("mnist")


def test_load_pretrained_rejects_registry_entry_without_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry entries without a matching weight schema fail before filesystem access."""
    monkeypatch.setitem(_pt._REGISTRY, "orphan", (mnist_classifier, "orphan.npz"))

    with pytest.raises(ValueError, match="missing weight schema"):
        _pt.load_pretrained("orphan")


def test_load_pretrained_rejects_projection_count_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Projection layout mismatches fail before any CSR weights are overwritten."""
    _builder, filename = _pt._REGISTRY["mnist"]
    _write_mnist_archive(tmp_path, filename)
    captured: dict[str, Network] = {}
    before_data: dict[str, np.ndarray[Any, Any]] = {}

    def build_with_missing_projection() -> Network:
        net = mnist_classifier()
        net.projections = net.projections[:1]
        captured["net"] = net
        before_data["projection0"] = np.array(net.projections[0].data, copy=True)
        return net

    monkeypatch.setattr(_pt, "_WEIGHTS_DIR", tmp_path)
    monkeypatch.setitem(_pt._REGISTRY, "mnist", (build_with_missing_projection, filename))

    with pytest.raises(ValueError, match="projection count"):
        _pt.load_pretrained("mnist")

    np.testing.assert_array_equal(captured["net"].projections[0].data, before_data["projection0"])


def test_load_pretrained_rejects_projection_topology_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Projection dimensions must match the archive schema before weight writes."""
    _builder, filename = _pt._REGISTRY["mnist"]
    _write_mnist_archive(tmp_path, filename)
    captured: dict[str, Network] = {}
    before_data: dict[str, np.ndarray[Any, Any]] = {}

    def build_with_wrong_projection_shape() -> Network:
        net = mnist_classifier()
        captured["net"] = net
        before_data["projection0"] = np.array(net.projections[0].data, copy=True)
        net.projections[0].source.n = 42
        return net

    monkeypatch.setattr(_pt, "_WEIGHTS_DIR", tmp_path)
    monkeypatch.setitem(_pt._REGISTRY, "mnist", (build_with_wrong_projection_shape, filename))

    with pytest.raises(ValueError, match="projection 0"):
        _pt.load_pretrained("mnist")

    np.testing.assert_array_equal(captured["net"].projections[0].data, before_data["projection0"])
