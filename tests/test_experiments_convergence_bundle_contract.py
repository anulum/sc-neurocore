# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for convergence demonstration bundle contract

"""Behavioural contract for convergence demonstration bundle output."""

from __future__ import annotations


def test_convergence_demonstration_writes_mlir_bundle(tmp_path) -> None:
    from sc_neurocore.experiments.demonstration_convergence import run_demonstration

    result = run_demonstration(
        hardware_mode="bundle",
        work_dir=str(tmp_path / "compiler"),
        bundle_dir=str(tmp_path / "mlir_bundle"),
    )

    assert result["hardware"]["mode"] == "bundle"
    assert result["hardware"]["verilog_path"] is None
    bundle = result["hardware"]["bundle"]
    assert bundle["module_name"] == "director_top"
    assert bundle["node_count"] >= 1
    assert (tmp_path / "mlir_bundle" / "director_top.mlir").is_file()
    assert (tmp_path / "mlir_bundle" / "mlir_bundle_manifest.json").is_file()
