# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evidence packet and toolchain division contract

"""The toolchain runs left the evidence packet module; every historical import still reaches them.

``sc_neurocore.safety_cert.evidence_packet`` held two responsibilities: the packet's format,
manifest and verification, and running the external toolchains it records. The runners and
their provenance now live in ``sc_neurocore.safety_cert.evidence_toolchain``. The packet module
re-exports the public names as the same objects, so code and pickles that name
``sc_neurocore.safety_cert.evidence_packet.run_formal_job`` keep working.
"""

from __future__ import annotations

import ast
import inspect
import pickle
from pathlib import Path

import sc_neurocore.safety_cert.evidence_packet as evidence_packet
import sc_neurocore.safety_cert.evidence_toolchain as evidence_toolchain

PUBLIC_TOOLCHAIN_NAMES = (
    "FormalTarget",
    "TOOLS",
    "checkout_root",
    "place_and_route",
    "run_formal_job",
    "source_identity",
    "synthesize",
    "tool_versions",
)


def test_historical_packet_imports_are_the_toolchain_module_objects() -> None:
    """Each public toolchain name of the packet module is the toolchain module's own object."""
    for name in PUBLIC_TOOLCHAIN_NAMES:
        assert getattr(evidence_packet, name) is getattr(evidence_toolchain, name), name
    assert set(PUBLIC_TOOLCHAIN_NAMES) - {"TOOLS"} <= set(evidence_packet.__all__)


def test_the_runners_are_defined_in_the_toolchain_module() -> None:
    """The runners and the job descriptor are owned by the toolchain module, by name and pickle."""
    for name in PUBLIC_TOOLCHAIN_NAMES:
        value = getattr(evidence_toolchain, name)
        if name == "TOOLS":
            continue
        assert value.__module__ == "sc_neurocore.safety_cert.evidence_toolchain", name
        assert value.__qualname__ == name
        assert pickle.loads(pickle.dumps(getattr(evidence_packet, name))) is value


def test_the_packet_module_defines_only_the_packet() -> None:
    """The packet's own definitions are its format, selection, boundary, export and checks."""
    tree = ast.parse(Path(inspect.getfile(evidence_packet)).read_text(encoding="utf-8"))
    defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert defined == {
        "PacketExport",
        "PacketVerification",
        "formal_targets",
        "boundary_statement",
        "_render_text",
        "_category",
        "_write_cosim",
        "export_evidence_packet",
        "_read_manifest",
        "verify_evidence_packet",
    }
    assert defined.isdisjoint(vars(evidence_toolchain))
