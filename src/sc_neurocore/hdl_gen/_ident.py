# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HDL identifier validation helpers

from __future__ import annotations

import re

_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")

_VERILOG_RESERVED = frozenset(
    {
        "module",
        "always",
        "begin",
        "end",
        "assign",
        "wire",
        "reg",
        "input",
        "output",
        "inout",
        "parameter",
        "localparam",
        "generate",
        "genvar",
        "if",
        "else",
        "case",
        "endcase",
        "for",
        "while",
        "function",
        "task",
        "automatic",
        "initial",
        "posedge",
        "negedge",
        "edge",
        "or",
        "and",
        "not",
        "xor",
        "nand",
        "nor",
        "xnor",
        "buf",
        "bufif0",
        "bufif1",
        "notif0",
        "notif1",
        "tran",
        "tranif0",
        "tranif1",
        "rtran",
        "rtranif0",
        "rtranif1",
        "supply0",
        "supply1",
        "strong0",
        "strong1",
        "pull0",
        "pull1",
        "weak0",
        "weak1",
        "highz0",
        "highz1",
        "wait",
        "disable",
        "fork",
        "join",
        "repeat",
        "forever",
        "default",
        "casex",
        "casez",
        "deassign",
        "force",
        "release",
        "primitive",
        "endprimitive",
        "table",
        "endtable",
        "specify",
        "endspecify",
        "endmodule",
        "endfunction",
        "endtask",
        "endgenerate",
        "integer",
        "real",
        "realtime",
        "time",
        "tri",
        "tri0",
        "tri1",
        "triand",
        "trior",
        "trireg",
        "wand",
        "wor",
        "scalared",
        "vectored",
        "signed",
        "unsigned",
    }
)


def sanitize_ident(name: str, context: str = "identifier") -> str:
    """Validate an HDL-facing identifier before interpolating it into source."""
    if not _IDENT_RE.fullmatch(name):
        raise ValueError(f"Invalid {context}: {name!r}")
    if name in _VERILOG_RESERVED:
        raise ValueError(f"Invalid {context}: {name!r}")
    return name
