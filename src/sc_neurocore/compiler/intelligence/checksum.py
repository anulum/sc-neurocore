# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Model checksum

"""Embed SHA-256 model checksums in Verilog source."""

from __future__ import annotations

import hashlib
import json


def embed_model_checksum(
    verilog: str,
    *,
    equations: dict[str, str] | None = None,
    params: dict[str, int | float] | None = None,
) -> str:
    """Embed a SHA-256 checksum of the compiled model in the Verilog source."""
    payload = {
        "equations": equations or {},
        "params": params or {},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(canonical.encode()).hexdigest()

    checksum_block = (
        f"// ── SC-NeuroCore Model Checksum ──────────────────────────────\n"
        f"// SHA-256: {sha}\n"
        f"// Source: {canonical[:80]}{'...' if len(canonical) > 80 else ''}\n"
        f"// Verify: echo -n '{canonical}' | sha256sum\n"
        f"localparam [255:0] MODEL_HASH = 256'h{sha};\n"
    )

    line_list = verilog.split("\n")
    insert_pos = 1
    for i, line in enumerate(line_list):
        if line.strip().startswith("module"):
            insert_pos = i
            break

    line_list.insert(insert_pos, checksum_block)
    return "\n".join(line_list)
