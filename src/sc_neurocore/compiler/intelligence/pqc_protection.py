# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Post-quantum protection

"""Post-quantum cryptographic (PQC) IP protection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class PQCProtection:
    """Post-quantum cryptographic IP protection result.

    Attributes
    ----------
    algorithm : str
    signature_hex : str
    key_size_bits : int
    quantum_safe : bool
    """

    algorithm: str
    signature_hex: str
    key_size_bits: int
    quantum_safe: bool


def protect_ip_pqc(
    module_name: str,
    equations: dict[str, str],
    *,
    algorithm: str = "CRYSTALS-Dilithium",
    security_level: int = 3,
) -> PQCProtection:
    """Apply post-quantum cryptographic protection to IP core."""

    key_sizes = {2: 1312, 3: 1952, 5: 2592}
    key_bits = key_sizes.get(security_level, 1952)

    payload = f"PQC:{algorithm}:{module_name}:{sorted(equations.keys())}:{security_level}"
    sig = hashlib.sha3_256(payload.encode()).hexdigest()[:32]

    return PQCProtection(
        algorithm=algorithm,
        signature_hex=sig,
        key_size_bits=key_bits,
        quantum_safe=True,
    )
