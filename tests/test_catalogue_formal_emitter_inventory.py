# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (emitter_inventory) from former test_catalogue_formal.py

"""Verify catalogue job inventory, source generation and solver retention."""

from __future__ import annotations

from tests.catalogue_formal_support import *  # noqa: F403


def test_emitter_lists_every_perfect_model() -> None:
    """Every perfect class has a generated or explicit curated formal lane."""
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
    mapped = set(mod.CLASS_TO_SCHEMA) | set(mod.CURATED_CLASS_TO_MODULE)
    missing = perfect - mapped
    assert not missing, f"perfect models missing from formal emitter: {sorted(missing)}"

    overlap = set(mod.CLASS_TO_SCHEMA) & set(mod.CURATED_CLASS_TO_MODULE)
    assert not overlap, f"classes cannot be both generated and curated: {sorted(overlap)}"


def test_catalogue_formal_inventory_matches_perfect_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain enrolled solvers and exact regenerated sources with and without licence headers."""
    import importlib.util
    import sys

    name = "emit_catalogue_formal_inventory"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    sby_jobs = sorted(CATALOGUE.glob("*.sby"))
    assert CATALOGUE.is_dir(), "hdl/formal/catalogue missing — run emit_catalogue_formal.py"
    assert len(sby_jobs) >= len(_perfect_class_names())
    committed_modules = {path.stem for path in sby_jobs}
    curated_perfect = set(mod.CURATED_CLASS_TO_MODULE.values())
    assert curated_perfect <= committed_modules
    for sby in sby_jobs:
        module = sby.stem
        assert (CATALOGUE / f"{module}.v").is_file(), f"missing RTL for {module}"
        assert (CATALOGUE / f"{module}_formal.v").is_file(), f"missing harness for {module}"
        text = sby.read_text(encoding="utf-8")
        assert "mode bmc" in text or "mode prove" in text
        assert "smtbmc " in text

    # Exercise the public emitter without overwriting the committed job. A solver
    # repair must survive regeneration of both the job and its inventory row.
    import json

    monkeypatch.setattr(mod, "OUT_DIR", tmp_path)
    inventory = json.loads((CATALOGUE / "inventory.json").read_text(encoding="utf-8"))
    for class_name in ("WangBuzsakiNeuron", "AdaptiveThresholdIFNeuron"):
        row = next(job for job in inventory["jobs"] if job["class"] == class_name)
        module = row["module"]
        shutil.copyfile(CATALOGUE / f"{module}.sby", tmp_path / f"{module}.sby")
        result = mod.emit_one(class_name)
        assert result.solver == row["solver"]
        for suffix in (".v", "_formal.v", ".sby"):
            name = result.module + suffix
            assert (tmp_path / name).read_bytes() == (CATALOGUE / name).read_bytes()


def test_mckean_uses_only_the_curated_q3232_formal_lane() -> None:
    """The source McKean identity must not acquire a competing generated core."""
    import importlib.util
    import sys

    name = "emit_catalogue_formal_mckean_routing"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    assert "McKeanNeuron" not in mod.CLASS_TO_SCHEMA
    assert mod.CURATED_CLASS_TO_MODULE["McKeanNeuron"] == "mckean"
    assert not list(CATALOGUE.glob("sc_mckeanneuron*"))
    for suffix in (".v", "_formal.v", ".sby"):
        assert (CATALOGUE / f"mckean{suffix}").is_file()
    assert "| McKeanNeuron |" not in (CATALOGUE / "INVENTORY.md").read_text(encoding="utf-8")


def test_catalogue_formal_rtl_is_equation_compiler_output() -> None:
    """Generic RTL is compiler output; curated RTL remains explicit and real."""
    import importlib.util
    import sys

    name = "emit_catalogue_formal_rtl"
    spec = importlib.util.spec_from_file_location(name, EMITTER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    for rtl in sorted(CATALOGUE.glob("sc_*.v")):
        if rtl.name.endswith("_formal.v"):
            continue
        text = rtl.read_text(encoding="utf-8")
        if rtl.stem in mod.CURATED_FORMAL_MODULES:
            assert "output " in text
        else:
            assert "Auto-generated by SC-NeuroCore equation compiler" in text
            assert "spike_out" in text
        assert "module " in text
