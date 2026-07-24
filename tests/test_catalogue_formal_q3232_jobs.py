# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (q3232_jobs) from former test_catalogue_formal.py

from __future__ import annotations

from tests.catalogue_formal_support import *  # noqa: F403

def test_expif_formal_job_uses_enrolled_q3232_precision() -> None:
    """Keep the formal job aligned with the proven ExpIF fixed-point envelope."""
    import importlib.util

    name = "emit_catalogue_formal_expif_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["ExpIFNeuron"] == "exp_if"
    assert module.PRECISION_BY_SCHEMA["exp_if"] == (64, 32)
    assert module.DEPTH_BY_SCHEMA["exp_if"] == 4
    assert "exp_if" in module.MINIMAL_SAFETY_SCHEMAS


def test_wong_wang_formal_job_uses_enrolled_q3232_bounded_safety() -> None:
    """Keep Wong-Wang formal emission inside its Q32.32 H1 evidence boundary."""
    import importlib.util

    name = "emit_catalogue_formal_wong_wang_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["WongWangUnit"] == "wong_wang"
    assert module.PRECISION_BY_SCHEMA["wong_wang"] == (64, 32)
    assert module.DEPTH_BY_SCHEMA["wong_wang"] == 4
    assert "wong_wang" in module.MINIMAL_SAFETY_SCHEMAS
    harness = (CATALOGUE / "sc_wongwangunit_formal.v").read_text(encoding="utf-8")
    assert "Minimal safety: async reset clears the spike flag" in harness
    assert "Saturation contract" not in harness


def test_jansen_rit_formal_job_uses_enrolled_q3232_bounded_safety() -> None:
    """Keep Jansen-Rit formal emission inside its Q32.32 H1 evidence boundary."""
    import importlib.util

    name = "emit_catalogue_formal_jansen_rit_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["JansenRitUnit"] == "jansen_rit"
    assert module.PRECISION_BY_SCHEMA["jansen_rit"] == (64, 32)
    assert module.DEPTH_BY_SCHEMA["jansen_rit"] == 4
    assert "jansen_rit" in module.MINIMAL_SAFETY_SCHEMAS
    harness = (CATALOGUE / "sc_jansenritunit_formal.v").read_text(encoding="utf-8")
    assert "Minimal safety: async reset clears the spike flag" in harness
    assert "Saturation contract" not in harness


def test_mpr_formal_job_uses_enrolled_q3232_bounded_safety() -> None:
    """Keep MPR formal emission inside its Q32.32 H1 evidence boundary."""
    import importlib.util

    name = "emit_catalogue_formal_mpr_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["ErmentroutKopellPopulation"] == "ermentrout_kopell_pop"
    assert module.PRECISION_BY_SCHEMA["ermentrout_kopell_pop"] == (64, 32)
    assert module.DEPTH_BY_SCHEMA["ermentrout_kopell_pop"] == 4
    assert "ermentrout_kopell_pop" in module.MINIMAL_SAFETY_SCHEMAS
    assert "ermentrout_kopell_pop" in module.EVENT_SILENT_SCHEMAS
    harness = (CATALOGUE / "sc_ermentroutkopellpopulation_formal.v").read_text(encoding="utf-8")
    assert "Minimal safety: async reset clears the spike flag" in harness
    assert "reg past_valid = 1'b0;" in harness
    assert "if (past_valid && rst_n)" in harness
    assert harness.count("assert (spike_out == 1'b0);") == 2
    assert "uut." not in harness
    assert "Saturation contract" not in harness


def test_resonate_and_fire_formal_job_is_q3232_reset_safety_only() -> None:
    """Keep Model40 formal evidence inside its H1 public-port boundary."""
    import importlib.util

    name = "emit_catalogue_formal_resonate_and_fire_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["ResonateAndFireNeuron"] == "resonate_fire"
    assert module.PRECISION_BY_SCHEMA["resonate_fire"] == (64, 32)
    assert module.DEPTH_BY_SCHEMA["resonate_fire"] == 4
    assert "resonate_fire" in module.MINIMAL_SAFETY_SCHEMAS
    assert "resonate_fire" not in module.EVENT_SILENT_SCHEMAS
    harness = (CATALOGUE / "sc_resonate_and_fire_formal.v").read_text(encoding="utf-8")
    assert "Minimal safety: async reset clears the spike flag" in harness
    assert harness.count("assert (spike_out == 1'b0);") == 1
    assert "past_valid" not in harness
    assert "uut." not in harness
    assert "Saturation contract" not in harness


def test_alpha_formal_job_is_q3232_reset_safety_only() -> None:
    """Keep Model42 formal evidence inside its H1 public-port boundary."""
    import importlib.util

    name = "emit_catalogue_formal_alpha_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["AlphaNeuron"] == "alpha"
    assert module.PRECISION_BY_SCHEMA["alpha"] == (64, 32)
    assert module.DEPTH_BY_SCHEMA["alpha"] == 4
    assert "alpha" in module.MINIMAL_SAFETY_SCHEMAS
    assert "alpha" not in module.EVENT_SILENT_SCHEMAS
    harness = (CATALOGUE / "sc_alpha_synapse_lif_formal.v").read_text(encoding="utf-8")
    assert "Minimal safety: async reset clears the spike flag" in harness
    assert harness.count("assert (spike_out == 1'b0);") == 1
    assert "past_valid" not in harness
    assert "uut." not in harness
    assert "Saturation contract" not in harness


@pytest.mark.parametrize(
    ("class_name", "schema", "precision", "module_name", "flatten"),
    (
        ("SigmoidRateNeuron", "sigmoid_rate", (64, 32), "sc_sigmoidrateneuron", False),
        (
            "ThresholdLinearRateNeuron",
            "threshold_linear_rate",
            (32, 16),
            "sc_thresholdlinearrateneuron",
            False,
        ),
        ("WilsonCowanUnit", "wilson_cowan", (64, 32), "sc_wilsoncowanunit", True),
    ),
)
def test_rate_formal_jobs_match_enrolled_cosim_precision_and_event_silence(
    class_name: str,
    schema: str,
    precision: tuple[int, int],
    module_name: str,
    flatten: bool,
) -> None:
    """Keep continuous-rate formal jobs within their enrolled H1 boundary."""
    import importlib.util

    name = f"emit_catalogue_formal_{schema}_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA[class_name] == schema
    assert module.PRECISION_BY_SCHEMA[schema] == precision
    assert module.DEPTH_BY_SCHEMA[schema] == 4
    assert schema in module.MINIMAL_SAFETY_SCHEMAS
    assert schema in module.EVENT_SILENT_SCHEMAS
    assert (schema in module.FLATTEN_FORMAL_SCHEMAS) is flatten

    harness = (CATALOGUE / f"{module_name}_formal.v").read_text(encoding="utf-8")
    assert harness.count("assert (spike_out == 1'b0);") == 2
    assert "Saturation contract" not in harness

    sby = (CATALOGUE / f"{module_name}.sby").read_text(encoding="utf-8")
    assert (" -flatten" in sby) is flatten


def test_adaptive_threshold_if_formal_job_is_q3232_reset_safety_only() -> None:
    """Keep Model41 formal evidence inside its H1 public-port boundary."""
    import importlib.util

    name = "emit_catalogue_formal_adaptive_threshold_if_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["AdaptiveThresholdIFNeuron"] == "adaptive_threshold_if"
    assert module.PRECISION_BY_SCHEMA["adaptive_threshold_if"] == (64, 32)
    assert module.DEPTH_BY_SCHEMA["adaptive_threshold_if"] == 4
    assert "adaptive_threshold_if" in module.MINIMAL_SAFETY_SCHEMAS
    assert "adaptive_threshold_if" not in module.EVENT_SILENT_SCHEMAS
    harness = (CATALOGUE / "sc_adaptive_threshold_if_formal.v").read_text(encoding="utf-8")
    assert "Minimal safety: async reset clears the spike flag" in harness
    assert harness.count("assert (spike_out == 1'b0);") == 1
    assert "past_valid" not in harness
    assert "uut." not in harness
    assert "Saturation contract" not in harness
