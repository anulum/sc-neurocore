# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (other_precision_jobs) from former test_catalogue_formal.py

from __future__ import annotations

from tests.catalogue_formal_support import *  # noqa: F403


def test_connor_stevens_formal_job_uses_enrolled_q1616_precision() -> None:
    """Keep committed Connor-Stevens RTL aligned with the bounded co-sim carrier."""
    import importlib.util

    name = "emit_catalogue_formal_connor_stevens_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["ConnorStevensNeuron"] == "connor_stevens"
    assert module.PRECISION_BY_SCHEMA["connor_stevens"] == (32, 16)
    assert module.DEPTH_BY_SCHEMA["connor_stevens"] == 4
    assert "connor_stevens" in module.MINIMAL_SAFETY_SCHEMAS
    assert module.FORMAL_FIXED_CURRENT_BY_SCHEMA["connor_stevens"] == 100.0


def test_fitzhugh_nagumo_formal_job_uses_enrolled_q1616_precision() -> None:
    """Keep formal FitzHugh-Nagumo RTL aligned with exact event co-simulation."""
    import importlib.util

    name = "emit_catalogue_formal_fitzhugh_nagumo_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["FitzHughNagumoNeuron"] == "fitzhugh_nagumo"
    assert module.PRECISION_BY_SCHEMA["fitzhugh_nagumo"] == (32, 16)
    assert module.DEPTH_BY_SCHEMA["fitzhugh_nagumo"] == 4
    assert "fitzhugh_nagumo" in module.MINIMAL_SAFETY_SCHEMAS


def test_dpi_formal_job_uses_enrolled_q1616_precision() -> None:
    """Keep formal DPI RTL aligned with its three-state co-simulation envelope."""
    import importlib.util

    name = "emit_catalogue_formal_dpi_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["DPINeuron"] == "dpi_neuron"
    assert module.PRECISION_BY_SCHEMA["dpi_neuron"] == (32, 16)
    assert module.DEPTH_BY_SCHEMA["dpi_neuron"] == 4
    assert "dpi_neuron" in module.MINIMAL_SAFETY_SCHEMAS


def test_coba_lif_formal_job_uses_enrolled_q2424_precision() -> None:
    """Keep formal COBA LIF RTL aligned with its four-state co-simulation envelope."""
    import importlib.util

    name = "emit_catalogue_formal_coba_lif_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["COBALIFNeuron"] == "coba_lif"
    assert module.PRECISION_BY_SCHEMA["coba_lif"] == (48, 24)
    assert module.DEPTH_BY_SCHEMA["coba_lif"] == 4
    assert "coba_lif" in module.MINIMAL_SAFETY_SCHEMAS


def test_escape_rate_formal_job_uses_seeded_q2424_precision() -> None:
    """Keep formal stochastic RTL aligned with the full-period co-simulation."""
    import importlib.util

    name = "emit_catalogue_formal_escape_rate_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["EscapeRateNeuron"] == "escape_rate"
    assert module.PRECISION_BY_SCHEMA["escape_rate"] == (48, 24)
    assert module.DEPTH_BY_SCHEMA["escape_rate"] == 4
    assert "escape_rate" in module.MINIMAL_SAFETY_SCHEMAS


def test_poisson_formal_job_supports_a_stateless_seeded_q2424_module() -> None:
    """Keep the spike-only Poisson job aligned with full-period co-simulation."""
    import importlib.util

    name = "emit_catalogue_formal_poisson_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["PoissonNeuron"] == "poisson"
    assert module.PRECISION_BY_SCHEMA["poisson"] == (48, 24)
    assert module.DEPTH_BY_SCHEMA["poisson"] == 4
    assert "poisson" in module.MINIMAL_SAFETY_SCHEMAS
    ports = module._parse_module_ports(
        (CATALOGUE / "sc_poissonneuron.v").read_text(encoding="utf-8")
    )
    assert ports.primary_state is None
    assert ports.signed_outputs == ()
    assert ports.bit_outputs == ("spike_out",)


def test_iqif_formal_job_uses_bit_true_q320_precision() -> None:
    """Keep the IQIF formal job on its exact signed-integer datapath."""
    import importlib.util

    name = "emit_catalogue_formal_iqif_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["IntegerQIFNeuron"] == "iqif"
    assert module.PRECISION_BY_SCHEMA["iqif"] == (32, 0)
    assert module.DEPTH_BY_SCHEMA["iqif"] == 4
    assert "iqif" not in module.MINIMAL_SAFETY_SCHEMAS
    ports = module._parse_module_ports(
        (CATALOGUE / "sc_integerqifneuron.v").read_text(encoding="utf-8")
    )
    assert ports.primary_state == "v_out"
    assert ports.signed_outputs == ("v_out",)
    assert ports.bit_outputs == ("spike_out",)


def test_mcculloch_pitts_formal_job_is_stateless_q320_safety() -> None:
    """Keep the count/sentinel rule on its exact signed integer carrier."""
    import importlib.util

    name = "emit_catalogue_formal_mcculloch_pitts_precision"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    assert module.CLASS_TO_SCHEMA["McCullochPittsNeuron"] == "mcculloch_pitts"
    assert module.PRECISION_BY_SCHEMA["mcculloch_pitts"] == (32, 0)
    assert module.DEPTH_BY_SCHEMA["mcculloch_pitts"] == 4
    assert "mcculloch_pitts" in module.MINIMAL_SAFETY_SCHEMAS
    ports = module._parse_module_ports(
        (CATALOGUE / "sc_mccullochpittsneuron.v").read_text(encoding="utf-8")
    )
    assert ports.primary_state is None
    assert ports.signed_outputs == ()
    assert ports.bit_outputs == ("spike_out",)
