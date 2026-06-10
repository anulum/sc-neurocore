# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bitstream encryption wrapper

"""Bitstream encryption TCL/constraints generation for secure boot."""

from __future__ import annotations


def generate_bitstream_encryption(
    module_name: str,
    *,
    vendor: str = "xilinx",
    key_length: int = 256,
    key_source: str = "efuse",
) -> str:
    """Generate bitstream encryption TCL/constraints for secure boot."""
    if vendor == "xilinx":
        lines = [
            f"# Bitstream encryption for {module_name}",
            f"# SC-NeuroCore — Xilinx AES-{key_length} secure boot",
            f"# Key source: {key_source}",
            "",
            "# ── Vivado TCL commands ──",
            "set_property BITSTREAM.ENCRYPTION.ENCRYPT YES [current_design]",
            f"set_property BITSTREAM.ENCRYPTION.ENCRYPTKEYSELECT {key_source.upper()} [current_design]",
            "set_property BITSTREAM.ENCRYPTION.KEYLIFE {{100}} [current_design]",
            "",
            "# ── Key file reference ──",
            f"# Generate key: write_bitstream -encrypt -encrypt_key_file {module_name}.nky",
            f"set_property BITSTREAM.ENCRYPTION.KEYFILE {{{module_name}.nky}} [current_design]",
            "",
            "# ── Tamper detection ──",
            "set_property BITSTREAM.CONFIG.USR_ACCESS TIMESTAMP [current_design]",
            "set_property BITSTREAM.CONFIG.SECURITY_LEVEL LEVEL2 [current_design]",
            "",
            "# ── Authentication (optional HMAC) ──",
            "# set_property BITSTREAM.AUTHENTICATION.AUTHENTICATE YES [current_design]",
            f"# set_property BITSTREAM.AUTHENTICATION.HMACKEY_FILE {{{module_name}.hmac}} [current_design]",
        ]
    else:  # Intel
        lines = [
            f"# Bitstream encryption for {module_name}",
            f"# SC-NeuroCore — Intel/Altera AES-{key_length} secure boot",
            f"# Key source: {key_source}",
            "",
            "# ── Quartus Settings ──",
            f'set_global_assignment -name ENCRYPTION_KEY_SOURCE "{key_source.upper()}"',
            f'set_global_assignment -name ENCRYPTION_SECURITY_KEY "{module_name}_key"',
            "set_global_assignment -name ENABLE_CONFIGURATION_BITSTREAM_ENCRYPTION ON",
            "",
            "# ── Anti-tamper ──",
            "set_global_assignment -name ENABLE_ANTI_TAMPER ON",
            'set_global_assignment -name ANTI_TAMPER_SCHEME "DETECT"',
            "",
            "# ── Secure device setup ──",
            f"# quartus_pgm --jtag --encrypt --key {module_name}.key",
        ]

    return "\n".join(lines)
