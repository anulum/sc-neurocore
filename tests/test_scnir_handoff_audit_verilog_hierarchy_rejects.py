# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCNIR Verilog hierarchy rejection contracts

"""Reject missing Verilog module declarations, ports, instances, and localparams."""

from pathlib import Path

import pytest

from tests.scnir_handoff_audit_support import (
    SCNIRHDLHandoffAuditError,
    _write_valid_handoff,
    audit_scnir_hdl_handoff,
)


def test_audit_rejects_hierarchy_module_without_declaration(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "mixed_audit_net_core.v").write_text("// empty\n", encoding="utf-8")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="does not declare module"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_hierarchy_module_missing_port(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    (handoff / "mixed_audit_net_core.v").write_text(
        "module mixed_audit_net_core;\nendmodule\n", encoding="utf-8"
    )
    with pytest.raises(SCNIRHDLHandoffAuditError, match="is missing port"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_top_instance_missing_port(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    top_path = handoff / "mixed_audit_net.v"
    top = top_path.read_text(encoding="utf-8").replace(
        "    .weight_i(mixed_audit_net_core__weight_i)\n", ""
    )
    top_path.write_text(top, encoding="utf-8")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="missing port"):
        audit_scnir_hdl_handoff(handoff)


def test_audit_rejects_missing_top_localparam(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    _write_valid_handoff(handoff)
    top_path = handoff / "mixed_audit_net.v"
    top = top_path.read_text(encoding="utf-8").replace(
        "localparam integer SCNIR_BITSTREAM_LENGTH = 512;\n", ""
    )
    top_path.write_text(top, encoding="utf-8")
    with pytest.raises(SCNIRHDLHandoffAuditError, match="top module missing"):
        audit_scnir_hdl_handoff(handoff)
