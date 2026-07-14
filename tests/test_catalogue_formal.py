# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Catalogue formal inventory and smoke proofs

"""Inventory and optional SymbiYosys smoke for dual-axis perfect models.

These tests drive the real emitted ``.sby`` / formal RTL under
``hdl/formal/catalogue/`` and the emitter tool — not a re-implemented harness.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "hdl" / "formal" / "catalogue"
EMITTER = ROOT / "tools" / "emit_catalogue_formal.py"


def _perfect_class_names() -> set[str]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    from sc_neurocore.neurons.descriptor_tiers import is_perfect
    from sc_neurocore.neurons.model_descriptor import parse_model_descriptor

    names: set[str] = set()
    desc_dir = ROOT / "src" / "sc_neurocore" / "neurons" / "model_descriptors"
    for path in desc_dir.glob("*.toml"):
        desc = parse_model_descriptor(tomllib.loads(path.read_text(encoding="utf-8")))
        if is_perfect(desc):
            names.add(desc.class_name)
    return names


def test_emitter_lists_every_perfect_model() -> None:
    """Emitter CLASS_TO_SCHEMA must cover every live dual-axis perfect class."""
    import importlib.util
    import sys

    name = "emit_catalogue_formal"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Dataclass evaluation requires the module to be registered first.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    perfect = _perfect_class_names()
    mapped = set(mod.CLASS_TO_SCHEMA)
    missing = perfect - mapped
    assert not missing, f"perfect models missing from formal emitter: {sorted(missing)}"


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


def test_catalogue_formal_inventory_matches_perfect_count() -> None:
    """Committed catalogue jobs equal the number of dual-axis perfect models."""
    sby_jobs = sorted(CATALOGUE.glob("*.sby"))
    assert CATALOGUE.is_dir(), "hdl/formal/catalogue missing — run emit_catalogue_formal.py"
    assert len(sby_jobs) == len(_perfect_class_names())
    for sby in sby_jobs:
        module = sby.stem
        assert (CATALOGUE / f"{module}.v").is_file(), f"missing RTL for {module}"
        assert (CATALOGUE / f"{module}_formal.v").is_file(), f"missing harness for {module}"
        text = sby.read_text(encoding="utf-8")
        assert "mode bmc" in text
        assert "smtbmc z3" in text


def test_catalogue_formal_rtl_is_equation_compiler_output() -> None:
    """Each RTL file is equation-compiler generated (not empty stub)."""
    for rtl in sorted(CATALOGUE.glob("sc_*.v")):
        if rtl.name.endswith("_formal.v"):
            continue
        text = rtl.read_text(encoding="utf-8")
        assert "Auto-generated by SC-NeuroCore equation compiler" in text
        assert "module " in text
        assert "spike_out" in text


@pytest.mark.parametrize(
    "sby_name",
    [
        "sc_lapicque.sby",
        "sc_perfect_integrator.sby",
        "sc_quadratic_if.sby",
        "sc_dpineuron.sby",
        "sc_poissonneuron.sby",
    ],
)
def test_catalogue_formal_smoke_pass(sby_name: str) -> None:
    """Run a compact subset of catalogue SymbiYosys jobs end-to-end."""
    if shutil.which("sby") is None:
        pytest.skip("sby not on PATH")
    if shutil.which("z3") is None and shutil.which("z3-solver") is None:
        pytest.skip("z3 not on PATH")
    sby_path = CATALOGUE / sby_name
    assert sby_path.is_file()
    proc = subprocess.run(
        ["sby", "-f", sby_name],
        cwd=CATALOGUE,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "DONE (PASS" in combined, combined[-2000:]
