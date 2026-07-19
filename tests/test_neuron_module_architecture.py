# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuron module ownership ratchets

"""Architecture ratchets for neuron implementations extracted from bucket files."""

import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_aihara_rust_implementation_is_owned_by_its_module() -> None:
    dedicated = ROOT / "engine/src/neurons/aihara_map.rs"
    bucket = ROOT / "engine/src/neurons/maps.rs"
    assert dedicated.is_file()
    assert "pub struct AiharaMapNeuron" in dedicated.read_text()
    assert "AiharaMap" not in bucket.read_text()
    assert sum(1 for _ in dedicated.open()) < 500


def test_legacy_rust_map_bucket_cannot_gain_models() -> None:
    """Freeze the legacy map bucket while models move to owned modules."""
    bucket = (ROOT / "engine/src/neurons/maps.rs").read_text()
    model_names = set(re.findall(r"^pub struct (\w+MapNeuron)", bucket, re.MULTILINE))
    assert model_names == {
        "CazellesMapNeuron",
        "ChialvoMapNeuron",
        "CourageNekorkinMapNeuron",
        "ErmentroutKopellMapNeuron",
        "IbarzTanakaMapNeuron",
        "KilincBhattMapNeuron",
        "MedvedevMapNeuron",
        "RulkovMapNeuron",
    }


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
    """Keep optional Julia state out of the package facade bucket."""
    runtime = ROOT / "src/sc_neurocore/accel/julia/neurons/_runtime.py"
    package_init = ROOT / "src/sc_neurocore/accel/julia/neurons/__init__.py"
    runtime_text = runtime.read_text()
    init_text = package_init.read_text()
    assert runtime.is_file()
    assert "def is_julia_error" in runtime_text
    assert "from juliacall import Main" in runtime_text
    assert "from juliacall import Main" not in init_text
    assert sum(1 for _ in runtime.open()) < 100
