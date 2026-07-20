# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuron module ownership ratchets

"""Architecture ratchets for dedicated neuron implementation modules."""

import ast
import inspect
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

_MODEL_CONTRACT_MODULES = {
    "test_model_adaptive_threshold_moe_contracts.py": "TestAdaptiveThresholdMoENeuron",
    "test_model_astrocyte_lif_contracts.py": "TestAstrocyteLIFNeuron",
    "test_model_cochlear_hair_cell_contracts.py": "TestCochlearHairCell",
    "test_model_dendritic_nmda_contracts.py": "TestDendriticNMDANeuron",
    "test_model_direction_selective_rgc_contracts.py": "TestDirectionSelectiveRGC",
    "test_model_hybrid_linear_attention.py": "TestHybridLinearAttentionNeuron",
    "test_model_multicompartment_mcn_contracts.py": "TestMulticompartmentMCNNeuron",
    "test_model_quantum_inspired_lif.py": "TestQuantumInspiredLIFNeuron",
    "test_short_term_plasticity_contracts.py": "TestShortTermPlasticitySynapse",
    "test_dopamine_stdp_contracts.py": "TestDopamineStdpSynapse",
    "test_triplet_stdp_contracts.py": "TestTripletSTDP",
}


def test_aihara_rust_implementation_is_owned_by_its_module() -> None:
    dedicated = ROOT / "engine/src/neurons/aihara_map.rs"
    aggregate = ROOT / "engine/src/neurons/maps.rs"
    assert dedicated.is_file()
    assert "pub struct AiharaMapNeuron" in dedicated.read_text()
    assert not aggregate.exists()
    assert sum(1 for _ in dedicated.open()) < 500


def test_rust_map_models_have_one_owned_module_each() -> None:
    """Each map model owns its implementation and focused Rust tests."""
    owners = {
        "cazelles_map.rs": "CazellesMapNeuron",
        "chialvo_map.rs": "ChialvoMapNeuron",
        "courage_nekorkin_map.rs": "CourageNekorkinMapNeuron",
        "ermentrout_kopell_map.rs": "ErmentroutKopellMapNeuron",
        "ibarz_tanaka_map.rs": "IbarzTanakaMapNeuron",
        "kilinc_bhatt_map.rs": "KilincBhattMapNeuron",
        "medvedev_map.rs": "MedvedevMapNeuron",
        "rulkov_map.rs": "RulkovMapNeuron",
    }
    neuron_root = ROOT / "engine/src/neurons"
    for filename, model_name in owners.items():
        source = (neuron_root / filename).read_text()
        definitions = re.findall(r"^pub struct (\w+MapNeuron)", source, re.MULTILINE)
        assert definitions == [model_name]
        assert "#[cfg(test)]" in source
        assert len(source.splitlines()) < 300


def test_rust_map_compatibility_namespace_has_no_implementation() -> None:
    """The legacy namespace remains re-export-only and file-free."""
    neuron_root = ROOT / "engine/src/neurons"
    module_source = (neuron_root / "mod.rs").read_text()
    assert not (neuron_root / "maps.rs").exists()
    compatibility = module_source.split("pub mod maps {", maxsplit=1)[1].split("}", maxsplit=1)[0]
    assert "pub use super::" in compatibility
    assert "pub struct" not in compatibility
    assert "impl " not in compatibility


def test_neuron_and_synapse_contracts_are_module_specific() -> None:
    """Mixed model test modules cannot replace per-model contract ownership."""
    test_root = ROOT / "tests"
    assert not (test_root / "test_gap_models.py").exists()
    for filename, expected_class in _MODEL_CONTRACT_MODULES.items():
        source = (test_root / filename).read_text()
        tree = ast.parse(source)
        classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        assert classes == [expected_class]
        assert len(source.splitlines()) < 300


def test_adaptive_threshold_julia_facade_is_not_in_package_init() -> None:
    dedicated = ROOT / "src/sc_neurocore/accel/julia/neurons/adaptive_threshold_if.py"
    package_init = ROOT / "src/sc_neurocore/accel/julia/neurons/__init__.py"
    dedicated_text = dedicated.read_text()
    init_text = package_init.read_text()
    assert "def simulate_adaptive_threshold_if" in dedicated_text
    assert "def simulate_adaptive_threshold_if" not in init_text
    assert "def _ensure_adaptive_threshold_if_loaded" not in init_text
    assert "simulate_adaptive_threshold_if as simulate_adaptive_threshold_if" in init_text
    assert sum(1 for _ in dedicated.open()) < 200


def test_alpha_julia_facade_is_not_in_package_init() -> None:
    from sc_neurocore.accel.julia import neurons
    from sc_neurocore.accel.julia.neurons import alpha

    assert neurons.simulate_alpha is alpha.simulate_alpha
    assert neurons._ensure_alpha_loaded is alpha._ensure_loaded
    implementation_path = inspect.getsourcefile(neurons.simulate_alpha)
    assert implementation_path is not None
    assert Path(implementation_path).resolve() == Path(alpha.__file__).resolve()


def test_julia_package_init_cannot_gain_model_implementations() -> None:
    """Freeze legacy facade bodies; new model code belongs in dedicated modules."""
    package_init = ROOT / "src/sc_neurocore/accel/julia/neurons/__init__.py"
    tree = ast.parse(package_init.read_text())
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    assert function_names == {
        "_as_ermentrout_kopell_pop_input",
        "_as_jansen_rit_input",
        "_as_resonate_and_fire_input",
        "_as_wilson_cowan_ext_input",
        "_as_wong_wang_inputs",
        "_dt_or_default",
        "_ensure_ermentrout_kopell_pop_loaded",
        "_ensure_jansen_rit_loaded",
        "_ensure_resonate_and_fire_loaded",
        "_ensure_rk4_neurons_loaded",
        "_ensure_wilson_cowan_loaded",
        "_ensure_wong_wang_loaded",
        "_normalise_model_name",
        "simulate_ermentrout_kopell_pop",
        "simulate_jansen_rit",
        "simulate_resonate_and_fire",
        "simulate_rk4_neuron",
        "simulate_wilson_cowan",
        "simulate_wong_wang",
    }


def test_julia_shared_runtime_is_owned_by_an_internal_module() -> None:
    """Keep optional Julia state out of the package facade."""
    runtime = ROOT / "src/sc_neurocore/accel/julia/neurons/_runtime.py"
    package_init = ROOT / "src/sc_neurocore/accel/julia/neurons/__init__.py"
    runtime_text = runtime.read_text()
    init_text = package_init.read_text()
    assert runtime.is_file()
    assert "def is_julia_error" in runtime_text
    assert "from juliacall import Main" in runtime_text
    assert "from juliacall import Main" not in init_text
    assert sum(1 for _ in runtime.open()) < 100
