# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Learning-bridge architecture tests

"""Facade identity, optional-dependency, and anti-GodFile contracts."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys

import pytest

from sc_neurocore._native import learning_bridge as bridge
from sc_neurocore._native import learning_factory, learning_runtime

from test_learning_bridge_support import FakeLearningLib
from tools.ci_workflow_inventory import ci_job_text

pytest_plugins = ("test_learning_bridge_support",)


def test_facade_preserves_historical_public_identity() -> None:
    """The facade keeps the historical public names as the same objects."""
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
    """The facade's runtime diagnostics are read-only views of the runtime."""
    assert bridge._lib is fake_learning_lib
    assert bridge._HAS_LEARNING is True
    bridge.set_deterministic_mode(12)
    assert bridge._DETERMINISTIC_SEED == 12
    assert bridge.is_available() is True
    with pytest.raises(AttributeError, match="no attribute"):
        bridge.__getattr__("not_a_real_attribute")


def test_factory_selects_all_backends(fake_learning_lib: FakeLearningLib) -> None:
    """The factory builds each supported backend."""
    rust = learning_factory.create_plasticity_layer(3, backend="RuSt")
    wgpu = learning_factory.create_plasticity_layer(3, backend="RUST-WGPU")
    assert isinstance(rust, bridge.RustRuleLayer)
    assert isinstance(wgpu, bridge.RustWgpuRuleLayer)
    if hasattr(bridge, "TorchRuleLayer"):
        torch_layer = learning_factory.create_plasticity_layer(3, backend="torch", autograd=False)
        assert isinstance(torch_layer, bridge.TorchRuleLayer)
    assert fake_learning_lib.layer_ptr and fake_learning_lib.wgpu_ptr


def test_factory_rejects_invalid_backend(fake_learning_lib: FakeLearningLib) -> None:
    """The factory refuses an unknown backend name."""
    del fake_learning_lib
    with pytest.raises(TypeError, match="string"):
        learning_factory.create_plasticity_layer(3, backend=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected"):
        learning_factory.create_plasticity_layer(3, backend="unknown")


def test_factory_reports_missing_torch_without_disabling_rust(
    monkeypatch: pytest.MonkeyPatch, fake_learning_lib: FakeLearningLib
) -> None:
    """A missing torch is reported without disabling the Rust backend."""
    key = "sc_neurocore._native.learning_torch"
    monkeypatch.setitem(sys.modules, key, None)
    with pytest.raises(ImportError, match="torch' extra"):
        learning_factory.create_plasticity_layer(3, backend="torch")
    assert isinstance(
        learning_factory.create_plasticity_layer(3, backend="rust"), bridge.RustRuleLayer
    )
    assert fake_learning_lib.layer_ptr


# The concern of each divided module, as the top-level classes and functions it owns.
# A new definition is placed by deciding which concern it belongs to, then listing it here.
_OWNED_DEFINITIONS: dict[str, frozenset[str]] = {
    "learning_factory.py": frozenset(
        {
            "create_plasticity_layer",
        }
    ),
    "learning_runtime.py": frozenset(
        {
            "OnlineO1SnapshotFFI",
            "_library_path",
            "_bind_rule_api",
            "_bind_learner_api",
            "_bind_layer_api",
            "_bind_wgpu_api",
            "_bind_online_o1_api",
            "_load_native_library",
            "_get_lib",
            "is_available",
            "has_symbol",
            "require_symbol",
            "destroy_noexcept",
            "set_deterministic_mode",
            "deterministic_seed",
        }
    ),
    "learning_rust.py": frozenset(
        {
            "_require_native",
            "_NativeOwner",
            "RustOnlineO1Synapse",
            "RustPlasticityRule",
            "RustEligentLearner",
        }
    ),
    "learning_rust_layer.py": frozenset(
        {
            "_validated_state",
            "RustRuleLayer",
        }
    ),
    "learning_torch.py": frozenset(
        {
            "TorchRuleLayer",
        }
    ),
    "learning_torch_autograd.py": frozenset(
        {
            "_BiologicalAutogradFactory",
        }
    ),
    "learning_torch_precision.py": frozenset(
        {
            "normalise_bit_spec",
            "normalise_clip",
            "quantise_tensor",
        }
    ),
    "learning_torch_support.py": frozenset(
        {
            "rule_parameters",
            "validate_input",
        }
    ),
    "learning_validation.py": frozenset(
        {
            "require_integral",
            "require_non_negative_integral",
            "require_integral_range",
            "require_count",
            "require_rule_type",
            "require_bool",
            "require_finite_float",
            "require_positive_float",
            "require_non_negative_float",
            "require_unit_interval",
            "require_u32_seed",
            "require_u64_seed",
            "saturate",
            "_require_vector",
            "as_bool_vector",
            "as_float_vector",
            "as_probability_vector",
        }
    ),
    "learning_wgpu.py": frozenset(
        {
            "RustWgpuRuleLayer",
        }
    ),
}

# The facade re-exports the divided modules and delegates; these are all it defines.
_FACADE_DEFINITIONS = frozenset({"is_available", "set_deterministic_mode", "__getattr__"})


def _top_level_definitions(path: Path) -> set[str]:
    """Return the names of the top-level classes and functions a module defines."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    kinds = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    return {node.name for node in tree.body if isinstance(node, kinds)}


def test_each_divided_module_owns_its_declared_concern_and_is_licensed() -> None:
    """Each divided module defines exactly its declared names, none twice, under the licence header."""
    native_dir = Path(bridge.__file__).parent
    paths = {path.name: path for path in native_dir.glob("learning_*.py")}
    assert set(paths) == set(_OWNED_DEFINITIONS) | {"learning_bridge.py"}
    owner: dict[str, str] = {}
    for name, declared in _OWNED_DEFINITIONS.items():
        assert _top_level_definitions(paths[name]) == declared, name
        for definition in declared:
            assert definition not in owner, (
                f"{definition} is defined in {owner.get(definition)} and {name}"
            )
            owner[definition] = name
        lines = paths[name].read_text(encoding="utf-8").splitlines()
        assert lines[0] == "# SPDX-License-Identifier: AGPL-3.0-or-later"
        assert any('"""' in line for line in lines[:20])


def test_facade_only_re_exports_and_delegates() -> None:
    """The facade defines no class and only its delegating functions; everything else is re-exported."""
    assert _top_level_definitions(Path(bridge.__file__)) == _FACADE_DEFINITIONS


def test_facade_private_loader_aliases_runtime() -> None:
    """The facade's private loader is the runtime's own loader."""
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
    workflow = ci_job_text("test")
    assert "Autonomous-learning bridge exact coverage" in workflow
    assert "COVERAGE_FILE=.coverage-learning" in workflow
    assert "--include='src/sc_neurocore/_native/learning*.py'" in workflow
    assert "-m pytest tests/test_native/test_learning_bridge_*.py -q" in workflow
    assert "--fail-under=100 --show-missing" in workflow
