# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Learning-bridge architecture tests

"""Facade identity, optional-dependency, and anti-GodFile contracts."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest

from sc_neurocore._native import learning_bridge as bridge
from sc_neurocore._native import learning_factory, learning_runtime

from test_learning_bridge_support import FakeLearningLib

pytest_plugins = ("test_learning_bridge_support",)


def test_facade_preserves_historical_public_identity() -> None:
    public_types = [
        bridge.OnlineO1SnapshotFFI,
        bridge.RustOnlineO1Synapse,
        bridge.RustPlasticityRule,
        bridge.RustEligentLearner,
        bridge.RustRuleLayer,
        bridge.RustWgpuRuleLayer,
    ]
    if hasattr(bridge, "TorchRuleLayer"):
        public_types.extend((bridge.TorchRuleLayer, bridge.AutogradSTDPLayer))
        assert bridge.AutogradSTDPLayer is bridge.TorchRuleLayer
    assert all(item.__module__ == bridge.__name__ for item in public_types)
    assert bridge.create_plasticity_layer.__module__ == bridge.__name__
    assert len(bridge.__all__) == len(set(bridge.__all__))


def test_facade_runtime_diagnostics_are_read_only_views(
    fake_learning_lib: FakeLearningLib,
) -> None:
    assert bridge._lib is fake_learning_lib
    assert bridge._HAS_LEARNING is True
    bridge.set_deterministic_mode(12)
    assert bridge._DETERMINISTIC_SEED == 12
    assert bridge.is_available() is True
    with pytest.raises(AttributeError, match="no attribute"):
        bridge.__getattr__("not_a_real_attribute")


def test_factory_selects_all_backends(fake_learning_lib: FakeLearningLib) -> None:
    rust = learning_factory.create_plasticity_layer(3, backend="RuSt")
    wgpu = learning_factory.create_plasticity_layer(3, backend="RUST-WGPU")
    assert isinstance(rust, bridge.RustRuleLayer)
    assert isinstance(wgpu, bridge.RustWgpuRuleLayer)
    if hasattr(bridge, "TorchRuleLayer"):
        torch_layer = learning_factory.create_plasticity_layer(3, backend="torch", autograd=False)
        assert isinstance(torch_layer, bridge.TorchRuleLayer)
    assert fake_learning_lib.layer_ptr and fake_learning_lib.wgpu_ptr


def test_factory_rejects_invalid_backend(fake_learning_lib: FakeLearningLib) -> None:
    del fake_learning_lib
    with pytest.raises(TypeError, match="string"):
        learning_factory.create_plasticity_layer(3, backend=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected"):
        learning_factory.create_plasticity_layer(3, backend="unknown")


def test_factory_reports_missing_torch_without_disabling_rust(
    monkeypatch: pytest.MonkeyPatch, fake_learning_lib: FakeLearningLib
) -> None:
    key = "sc_neurocore._native.learning_torch"
    monkeypatch.setitem(sys.modules, key, None)
    with pytest.raises(ImportError, match="torch' extra"):
        learning_factory.create_plasticity_layer(3, backend="torch")
    assert isinstance(
        learning_factory.create_plasticity_layer(3, backend="rust"), bridge.RustRuleLayer
    )
    assert fake_learning_lib.layer_ptr


def test_split_modules_remain_bounded_and_licensed() -> None:
    native_dir = Path(bridge.__file__).parent
    paths = sorted(native_dir.glob("learning_*.py"))
    assert paths
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 300, f"{path.name} grew to {len(lines)} lines"
        assert lines[0] == "# SPDX-License-Identifier: AGPL-3.0-or-later"
        assert any('"""' in line for line in lines[:20])


def test_facade_is_thin_relative_to_historical_monolith() -> None:
    facade_lines = Path(bridge.__file__).read_text(encoding="utf-8").splitlines()
    assert len(facade_lines) < 120
    assert not any(line.startswith("class Rust") for line in facade_lines)


def test_facade_private_loader_aliases_runtime() -> None:
    assert bridge._get_lib is learning_runtime._get_lib
    assert bridge._load_native_library is learning_runtime._load_native_library


def test_facade_imports_without_optional_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The historical facade remains usable when importing Torch fails."""
    monkeypatch.setitem(sys.modules, "torch", None)
    try:
        without_torch = importlib.reload(bridge)
        assert "TorchRuleLayer" not in without_torch.__all__
        assert "AutogradSTDPLayer" not in without_torch.__all__
    finally:
        monkeypatch.undo()
        importlib.reload(bridge)


def test_ci_enforces_exact_learning_bridge_coverage() -> None:
    """Python 3.12 CI keeps the modular native bridge at exact branch coverage."""
    root = Path(bridge.__file__).resolve().parents[3]
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "Autonomous-learning bridge exact coverage" in workflow
    assert "COVERAGE_FILE=.coverage-learning" in workflow
    assert "--include='src/sc_neurocore/_native/learning*.py'" in workflow
    assert "-m pytest tests/test_native/test_learning_bridge_*.py -q" in workflow
    assert "--fail-under=100 --show-missing" in workflow
