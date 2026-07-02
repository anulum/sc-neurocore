# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Host driver generation

"""Host driver generation utilities for compiled neuron modules.

Auto-generates C/Python drivers for memory-mapped bus wrappers.
"""

from __future__ import annotations

import re
from typing import Literal, NamedTuple

from .live_control import MMIOUpdateSpec


class _ParameterBinding(NamedTuple):
    """Sanitized generated-driver identifiers for one module parameter."""

    source_name: str
    register_identifier: str
    setter_identifier: str


def generate_host_driver(
    module_name: str,
    params: dict[str, int],
    *,
    language: Literal["python", "c"] = "python",
    bus: Literal["axi_lite", "wishbone"] = "axi_lite",
    base_address: int = 0x4000_0000,
    data_width: int = 16,
    fraction: int = 8,
    live_update_spec: MMIOUpdateSpec | None = None,
) -> str:
    """Generate host-side driver code for a bus-wrapped neuron.

    Parameters
    ----------
    module_name : str
        Neuron module name.
    params : dict[str, int]
        Parameter names and bit widths.
    language : str
        ``"python"`` or ``"c"``.
    bus : str
        Bus protocol.
    base_address : int
        Memory-mapped base address.
    data_width : int
        Fixed-point data width.
    fraction : int
        Fractional bits.
    live_update_spec : MMIOUpdateSpec, optional
        Live-control bank contract. When provided, generated drivers include
        CRC-guarded live-parameter update, trap check, and committed readback
        helpers for each named bank entry.

    Returns
    -------
    str
        Complete driver source code.
    """
    module_identifier = _module_identifier(module_name)
    if language == "python":
        return _gen_python_driver(
            module_identifier,
            params,
            base_address,
            data_width,
            fraction,
            live_update_spec,
        )
    elif language == "c":
        return _gen_c_driver(
            module_identifier,
            params,
            base_address,
            data_width,
            fraction,
            live_update_spec,
        )
    raise ValueError(f"Unsupported language: {language!r}")


def _driver_identifier(value: str) -> str:
    """Return a safe generated-driver identifier fragment."""
    safe = re.sub(r"[^0-9a-zA-Z_]+", "_", value.strip()).strip("_").lower()
    if not safe:
        raise ValueError("generated driver identifier must not be empty")
    if safe[0].isdigit():
        safe = f"p_{safe}"
    return safe


def _module_identifier(module_name: str) -> str:
    """Return the safe generated-driver identifier for a module name."""
    try:
        return _driver_identifier(module_name)
    except ValueError as exc:
        msg = "module_name must contain at least one alphanumeric identifier character"
        raise ValueError(msg) from exc


def _parameter_identifier(parameter_name: str) -> str:
    """Return the safe register identifier for a generated parameter."""
    return _driver_identifier(parameter_name)


def _setter_identifier(parameter_name: str) -> str:
    """Return the safe Python/C setter suffix for a generated parameter."""
    parameter_identifier = _parameter_identifier(parameter_name)
    has_symbolic_parameter_prefix = re.match(
        r"^\s*p(?:[^0-9a-zA-Z]+|_+)",
        parameter_name,
        flags=re.IGNORECASE,
    )
    if (
        has_symbolic_parameter_prefix is not None
        and parameter_identifier.startswith("p_")
        and len(parameter_identifier) > 2
        and not parameter_identifier[2].isdigit()
    ):
        return parameter_identifier[2:]
    return parameter_identifier


def _parameter_bindings(params: dict[str, int]) -> list[_ParameterBinding]:
    """Return collision-free generated identifiers for all parameters."""
    bindings: list[_ParameterBinding] = []
    seen: dict[str, str] = {}
    for source_name in params:
        register_identifier = _parameter_identifier(source_name)
        setter_identifier = _setter_identifier(source_name)
        for category, identifier in (
            ("register", register_identifier),
            ("setter", setter_identifier),
        ):
            generated_key = f"{category}:{identifier}"
            existing_source = seen.get(generated_key)
            if existing_source is not None:
                msg = (
                    "parameter identifier collision after sanitization: "
                    f"{existing_source!r} and {source_name!r} both map to {identifier!r}"
                )
                raise ValueError(msg)
            seen[generated_key] = source_name
        bindings.append(
            _ParameterBinding(
                source_name=source_name,
                register_identifier=register_identifier,
                setter_identifier=setter_identifier,
            )
        )
    return bindings


def _python_class_name(module_identifier: str) -> str:
    """Return the generated Python driver class name."""
    return "".join(part.capitalize() for part in module_identifier.split("_")) + "Driver"


def _gen_python_driver(
    module_identifier: str,
    params: dict[str, int],
    base_address: int,
    data_width: int,
    fraction: int,
    live_update_spec: MMIOUpdateSpec | None,
) -> str:
    """Generate Python MMIO driver."""
    class_name = _python_class_name(module_identifier)
    param_bindings = _parameter_bindings(params)
    lines = [
        f'"""Auto-generated Python driver for {module_identifier}.',
        "",
        "SC-NeuroCore deployment utilities.",
        f"Bus: memory-mapped I/O at 0x{base_address:08X}.",
        f"Fixed-point: Q{data_width - fraction - 1}.{fraction} ({data_width}-bit signed).",
        '"""',
        "",
        "from __future__ import annotations",
        "",
    ]
    if live_update_spec is not None:
        lines.extend(["import struct", "import zlib", ""])

    lines.extend(
        [
            "",
            f"class {class_name}:",
            f'    """Memory-mapped driver for {module_identifier}."""',
            "",
            f"    BASE = 0x{base_address:08X}",
            f"    FRACTION = {fraction}",
            "",
            "    # Register offsets",
            "    REG_CTRL        = 0x00  # bit0=enable, bit1=reset",
            "    REG_I_T         = 0x04  # Input current (Q-format)",
            "    REG_SPIKE_COUNT = 0x08  # Spike counter (read-only)",
        ]
    )

    for i, binding in enumerate(param_bindings):
        register_name = binding.register_identifier.upper()
        lines.append(f"    REG_{register_name:16s}= 0x{0x0C + i * 4:02X}")

    if live_update_spec is not None:
        addresses = live_update_spec.control_register_addresses
        lines.extend(
            [
                "",
                "    # Live-control register offsets",
                f"    LIVE_REG_CONTROL        = 0x{addresses['control']:02X}",
                f"    LIVE_REG_STATUS         = 0x{addresses['status']:02X}",
                f"    LIVE_REG_BANK_SELECT    = 0x{addresses['bank_select']:02X}",
                f"    LIVE_REG_ENTRY_INDEX    = 0x{addresses['entry_index']:02X}",
                f"    LIVE_REG_WRITE_DATA_LO  = 0x{addresses['write_data_lo']:02X}",
                f"    LIVE_REG_WRITE_DATA_HI  = 0x{addresses['write_data_hi']:02X}",
                f"    LIVE_REG_TRAP_STATUS    = 0x{addresses['trap_status']:02X}",
                f"    LIVE_REG_TRAP_CLEAR     = 0x{addresses['trap_clear']:02X}",
                f"    LIVE_REG_WRITE_CHECKSUM = 0x{addresses['write_checksum']:02X}",
                f"    LIVE_REG_READ_DATA_LO   = 0x{addresses['read_data_lo']:02X}",
                f"    LIVE_REG_READ_DATA_HI   = 0x{addresses['read_data_hi']:02X}",
                f"    LIVE_CTRL_UPDATE_VALID  = 0x{live_update_spec.control_bits['update_valid']:08X}",
                f"    LIVE_CTRL_COMMIT        = 0x{live_update_spec.control_bits['commit']:08X}",
                f"    LIVE_CTRL_ROLLBACK      = 0x{live_update_spec.control_bits['rollback']:08X}",
                f"    LIVE_CTRL_CLEAR_TRAP    = 0x{live_update_spec.control_bits['clear_trap']:08X}",
                f"    LIVE_STATUS_TRAP_LATCHED = 0x{live_update_spec.status_bits['trap_latched']:08X}",
                f"    LIVE_TRAP_CLEAR_MASK    = 0x{live_update_spec.trap_clear_mask:08X}",
            ]
        )
        for bank_index, bank in enumerate(live_update_spec.banks):
            bank_id = _driver_identifier(bank.bank_name).upper()
            lines.append(f"    LIVE_BANK_{bank_id} = {bank_index}")
            for parameter in bank.parameter_names:
                live_param_id = _driver_identifier(parameter).upper()
                lines.append(
                    f"    LIVE_{bank_id}_{live_param_id}_INDEX = {bank.entry_index(parameter)}"
                )
                lines.append(
                    f"    LIVE_{bank_id}_{live_param_id}_WIDTH_BITS = {bank.entry_width_bits}"
                )

    lines.extend(
        [
            "",
            "    def __init__(self, read_fn, write_fn, base: int = BASE):",
            '        """Initialise with platform-specific read/write functions.',
            "",
            "        Parameters",
            "        ----------",
            "        read_fn : callable",
            "            ``read_fn(addr) -> int`` — read 32-bit register.",
            "        write_fn : callable",
            "            ``write_fn(addr, value)`` — write 32-bit register.",
            "        base : int",
            "            Base address override.",
            '        """',
            "        self._read = read_fn",
            "        self._write = write_fn",
            "        self._base = base",
            "",
            "    def _wr(self, offset: int, value: int) -> None:",
            '        """Write a register."""',
            "        self._write(self._base + offset, value & 0xFFFFFFFF)",
            "",
            "    def _rd(self, offset: int) -> int:",
            '        """Read a register."""',
            "        return self._read(self._base + offset)",
            "",
        ]
    )

    if live_update_spec is not None:
        lines.extend(
            [
                "    def _live_crc32(self, bank_select: int, entry_index: int, data_lo: int, data_hi: int) -> int:",
                '        """Return IEEE CRC32 over the live-control update tuple."""',
                "        payload = struct.pack(",
                '            "<IIII",',
                "            bank_select & 0xFFFFFFFF,",
                "            entry_index & 0xFFFFFFFF,",
                "            data_lo & 0xFFFFFFFF,",
                "            data_hi & 0xFFFFFFFF,",
                "        )",
                "        return zlib.crc32(payload) & 0xFFFFFFFF",
                "",
                "    def _live_update_encoded(self, bank_select: int, entry_index: int, width_bits: int, encoded_word: int) -> None:",
                '        """CRC-stage, commit, and trap-check one encoded live coefficient."""',
                "        if not isinstance(encoded_word, int) or isinstance(encoded_word, bool):",
                '            raise ValueError("encoded_word must be an integer")',
                "        if encoded_word < 0 or encoded_word >= (1 << width_bits):",
                '            raise ValueError("encoded_word does not fit the live parameter width")',
                "        data_lo = encoded_word & 0xFFFFFFFF",
                "        data_hi = (encoded_word >> 32) & 0xFFFFFFFF",
                "        checksum = self._live_crc32(bank_select, entry_index, data_lo, data_hi)",
                "        self._wr(self.LIVE_REG_BANK_SELECT, bank_select)",
                "        self._wr(self.LIVE_REG_ENTRY_INDEX, entry_index)",
                "        self._wr(self.LIVE_REG_WRITE_DATA_LO, data_lo)",
                "        self._wr(self.LIVE_REG_WRITE_DATA_HI, data_hi)",
                "        self._wr(self.LIVE_REG_WRITE_CHECKSUM, checksum)",
                "        self._wr(self.LIVE_REG_CONTROL, self.LIVE_CTRL_UPDATE_VALID)",
                "        self._wr(self.LIVE_REG_CONTROL, self.LIVE_CTRL_COMMIT)",
                "        status = self._rd(self.LIVE_REG_STATUS)",
                "        if status & self.LIVE_STATUS_TRAP_LATCHED:",
                '            raise RuntimeError("live-control update latched a hardware trap")',
                "",
                "    def _live_read_encoded(self, bank_select: int, entry_index: int, width_bits: int) -> int:",
                '        """Read one committed active live-control coefficient."""',
                "        self._wr(self.LIVE_REG_BANK_SELECT, bank_select)",
                "        self._wr(self.LIVE_REG_ENTRY_INDEX, entry_index)",
                "        data_lo = self._rd(self.LIVE_REG_READ_DATA_LO) & 0xFFFFFFFF",
                "        data_hi = self._rd(self.LIVE_REG_READ_DATA_HI) & 0xFFFFFFFF if width_bits > 32 else 0",
                "        return ((data_hi << 32) | data_lo) & ((1 << width_bits) - 1)",
                "",
                "    def read_live_status(self) -> int:",
                '        """Read live-control status bits."""',
                "        return self._rd(self.LIVE_REG_STATUS)",
                "",
                "    def read_live_trap_status(self) -> int:",
                '        """Read sticky live-control trap status bits."""',
                "        return self._rd(self.LIVE_REG_TRAP_STATUS)",
                "",
                "    def rollback_live_shadow(self) -> None:",
                '        """Rollback loaded live-control shadow state from active coefficients."""',
                "        self._wr(self.LIVE_REG_CONTROL, self.LIVE_CTRL_ROLLBACK)",
                "",
                "    def clear_selected_live_traps(self, trap_mask: int) -> None:",
                '        """Clear selected sticky live-control trap bits."""',
                "        if not isinstance(trap_mask, int) or isinstance(trap_mask, bool):",
                '            raise ValueError("trap_mask must be an integer")',
                "        if trap_mask < 0 or trap_mask > self.LIVE_TRAP_CLEAR_MASK:",
                '            raise ValueError("trap_mask must select only live-control trap bits")',
                "        self._wr(self.LIVE_REG_TRAP_CLEAR, trap_mask)",
                "        self._wr(self.LIVE_REG_CONTROL, self.LIVE_CTRL_CLEAR_TRAP)",
                "",
                "    def clear_live_traps(self) -> None:",
                '        """Clear all sticky live-control trap bits."""',
                "        self.clear_selected_live_traps(self.LIVE_TRAP_CLEAR_MASK)",
                "",
            ]
        )

    lines.extend(
        [
            "    def encode_q(self, value: float) -> int:",
            '        """Encode a float to Q-format integer."""',
            "        return int(round(value * (1 << self.FRACTION)))",
            "",
            "    def decode_q(self, raw: int) -> float:",
            '        """Decode a Q-format integer to float."""',
            "        return raw / (1 << self.FRACTION)",
            "",
            "    # ── Control ─────────────────────────────────────────────",
            "",
            "    def enable(self) -> None:",
            '        """Enable the neuron (start clocking)."""',
            "        self._wr(self.REG_CTRL, 0x01)",
            "",
            "    def disable(self) -> None:",
            '        """Disable the neuron (stop clocking)."""',
            "        self._wr(self.REG_CTRL, 0x00)",
            "",
            "    def reset(self) -> None:",
            '        """Assert reset, then release."""',
            "        self._wr(self.REG_CTRL, 0x02)",
            "        self._wr(self.REG_CTRL, 0x01)",
            "",
            "    # ── I/O ─────────────────────────────────────────────────",
            "",
            "    def set_current(self, I: float) -> None:",
            '        """Set the input current."""',
            "        self._wr(self.REG_I_T, self.encode_q(I))",
            "",
            "    def get_spike_count(self) -> int:",
            '        """Read the spike counter."""',
            "        return self._rd(self.REG_SPIKE_COUNT)",
            "",
            "    # ── Parameters ──────────────────────────────────────────",
        ]
    )

    for binding in param_bindings:
        register_name = binding.register_identifier.upper()
        lines.extend(
            [
                "",
                f"    def set_{binding.setter_identifier}(self, value: float) -> None:",
                f'        """Set register {register_name}."""',
                f"        self._wr(self.REG_{register_name}, self.encode_q(value))",
            ]
        )

    if live_update_spec is not None:
        lines.extend(["", "    # -- Live-control parameter banks -------------------------------"])
        for bank in live_update_spec.banks:
            bank_id = _driver_identifier(bank.bank_name)
            bank_const = bank_id.upper()
            for parameter in bank.parameter_names:
                param_id = _driver_identifier(parameter)
                param_const = param_id.upper()
                suffix = f"{bank_id}_{param_id}"
                lines.extend(
                    [
                        "",
                        f"    def update_live_{suffix}_encoded(self, encoded_word: int) -> None:",
                        f'        """Update committed active parameter {bank_id}.{param_id}."""',
                        "        self._live_update_encoded(",
                        f"            self.LIVE_BANK_{bank_const},",
                        f"            self.LIVE_{bank_const}_{param_const}_INDEX,",
                        f"            self.LIVE_{bank_const}_{param_const}_WIDTH_BITS,",
                        "            encoded_word,",
                        "        )",
                        "",
                        f"    def read_live_{suffix}_encoded(self) -> int:",
                        f'        """Read committed active parameter {bank_id}.{param_id}."""',
                        "        return self._live_read_encoded(",
                        f"            self.LIVE_BANK_{bank_const},",
                        f"            self.LIVE_{bank_const}_{param_const}_INDEX,",
                        f"            self.LIVE_{bank_const}_{param_const}_WIDTH_BITS,",
                        "        )",
                        "",
                        f"    def verify_live_{suffix}_encoded(self, encoded_word: int) -> bool:",
                        f'        """Update {bank_id}.{param_id} and verify committed readback."""',
                        f"        self.update_live_{suffix}_encoded(encoded_word)",
                        f"        return self.read_live_{suffix}_encoded() == encoded_word",
                    ]
                )

    lines.append("")
    return "\n".join(lines)


def _gen_c_driver(
    module_identifier: str,
    params: dict[str, int],
    base_address: int,
    data_width: int,
    fraction: int,
    live_update_spec: MMIOUpdateSpec | None,
) -> str:
    """Generate C MMIO driver header."""
    module_macro = module_identifier.upper()
    guard = module_macro + "_DRIVER_H"
    param_bindings = _parameter_bindings(params)
    lines = [
        f"/* Auto-generated C driver for {module_identifier} */",
        "/* SC-NeuroCore deployment utilities */",
        f"/* Bus: MMIO at 0x{base_address:08X} */",
        "",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdint.h>",
        "",
        f"#define {module_macro}_BASE       0x{base_address:08X}U",
        f"#define {module_macro}_FRACTION    {fraction}",
        "",
        "/* Register offsets */",
        "#define REG_CTRL        0x00",
        "#define REG_I_T         0x04",
        "#define REG_SPIKE_COUNT 0x08",
    ]

    for i, binding in enumerate(param_bindings):
        lines.append(f"#define REG_{binding.register_identifier.upper():16s} 0x{0x0C + i * 4:02X}")

    if live_update_spec is not None:
        addresses = live_update_spec.control_register_addresses
        lines.extend(
            [
                "",
                "/* Live-control register offsets */",
                f"#define LIVE_REG_CONTROL         0x{addresses['control']:02X}",
                f"#define LIVE_REG_STATUS          0x{addresses['status']:02X}",
                f"#define LIVE_REG_BANK_SELECT     0x{addresses['bank_select']:02X}",
                f"#define LIVE_REG_ENTRY_INDEX     0x{addresses['entry_index']:02X}",
                f"#define LIVE_REG_WRITE_DATA_LO   0x{addresses['write_data_lo']:02X}",
                f"#define LIVE_REG_WRITE_DATA_HI   0x{addresses['write_data_hi']:02X}",
                f"#define LIVE_REG_TRAP_STATUS     0x{addresses['trap_status']:02X}",
                f"#define LIVE_REG_TRAP_CLEAR      0x{addresses['trap_clear']:02X}",
                f"#define LIVE_REG_WRITE_CHECKSUM  0x{addresses['write_checksum']:02X}",
                f"#define LIVE_REG_READ_DATA_LO    0x{addresses['read_data_lo']:02X}",
                f"#define LIVE_REG_READ_DATA_HI    0x{addresses['read_data_hi']:02X}",
                f"#define LIVE_CTRL_UPDATE_VALID   0x{live_update_spec.control_bits['update_valid']:08X}U",
                f"#define LIVE_CTRL_COMMIT         0x{live_update_spec.control_bits['commit']:08X}U",
                f"#define LIVE_CTRL_ROLLBACK       0x{live_update_spec.control_bits['rollback']:08X}U",
                f"#define LIVE_CTRL_CLEAR_TRAP     0x{live_update_spec.control_bits['clear_trap']:08X}U",
                f"#define LIVE_STATUS_TRAP_LATCHED 0x{live_update_spec.status_bits['trap_latched']:08X}U",
                f"#define LIVE_TRAP_CLEAR_MASK     0x{live_update_spec.trap_clear_mask:08X}U",
            ]
        )
        for bank_index, bank in enumerate(live_update_spec.banks):
            bank_id = _driver_identifier(bank.bank_name).upper()
            lines.append(f"#define LIVE_BANK_{bank_id} {bank_index}U")
            for parameter in bank.parameter_names:
                param_id = _driver_identifier(parameter).upper()
                lines.append(
                    f"#define LIVE_{bank_id}_{param_id}_INDEX {bank.entry_index(parameter)}U"
                )
                # Correction: usage bitwidth is sometimes larger than encoded width
                lines.append(
                    f"#define LIVE_{bank_id}_{param_id}_WIDTH_BITS {bank.entry_width_bits}U"
                )

    lines.extend(
        [
            "",
            "/* Platform-specific MMIO (user must implement) */",
            "extern void     mmio_write(uint32_t addr, uint32_t val);",
            "extern uint32_t mmio_read(uint32_t addr);",
            "",
        ]
    )

    if live_update_spec is not None:
        lines.extend(
            [
                "static inline uint32_t live_crc32_update_word(uint32_t crc_in, uint32_t data_word) {",
                "    uint32_t crc = crc_in;",
                "    for (uint32_t bit_idx = 0U; bit_idx < 32U; bit_idx++) {",
                "        if ((crc ^ (data_word >> bit_idx)) & 1U) {",
                "            crc = (crc >> 1U) ^ 0xEDB88320U;",
                "        } else {",
                "            crc >>= 1U;",
                "        }",
                "    }",
                "    return crc;",
                "}",
                "",
                "static inline uint32_t live_update_crc32(uint32_t bank_select, uint32_t entry_index, uint32_t data_lo, uint32_t data_hi) {",
                "    uint32_t crc = 0xFFFFFFFFU;",
                "    crc = live_crc32_update_word(crc, bank_select);",
                "    crc = live_crc32_update_word(crc, entry_index);",
                "    crc = live_crc32_update_word(crc, data_lo);",
                "    crc = live_crc32_update_word(crc, data_hi);",
                "    return crc ^ 0xFFFFFFFFU;",
                "}",
                "",
                "static inline int live_update_encoded(uint32_t bank_select, uint32_t entry_index, uint32_t width_bits, uint64_t encoded_word) {",
                "    if (width_bits < 64U && encoded_word >= (1ULL << width_bits)) {",
                "        return -1;",
                "    }",
                "    uint32_t data_lo = (uint32_t)(encoded_word & 0xFFFFFFFFULL);",
                "    uint32_t data_hi = (uint32_t)((encoded_word >> 32U) & 0xFFFFFFFFULL);",
                "    uint32_t checksum = live_update_crc32(bank_select, entry_index, data_lo, data_hi);",
                f"    mmio_write({module_macro}_BASE + LIVE_REG_BANK_SELECT, bank_select);",
                f"    mmio_write({module_macro}_BASE + LIVE_REG_ENTRY_INDEX, entry_index);",
                f"    mmio_write({module_macro}_BASE + LIVE_REG_WRITE_DATA_LO, data_lo);",
                f"    mmio_write({module_macro}_BASE + LIVE_REG_WRITE_DATA_HI, data_hi);",
                f"    mmio_write({module_macro}_BASE + LIVE_REG_WRITE_CHECKSUM, checksum);",
                f"    mmio_write({module_macro}_BASE + LIVE_REG_CONTROL, LIVE_CTRL_UPDATE_VALID);",
                f"    mmio_write({module_macro}_BASE + LIVE_REG_CONTROL, LIVE_CTRL_COMMIT);",
                f"    return (mmio_read({module_macro}_BASE + LIVE_REG_STATUS) & LIVE_STATUS_TRAP_LATCHED) ? -2 : 0;",
                "}",
                "",
                "static inline uint64_t live_read_encoded(uint32_t bank_select, uint32_t entry_index, uint32_t width_bits) {",
                f"    mmio_write({module_macro}_BASE + LIVE_REG_BANK_SELECT, bank_select);",
                f"    mmio_write({module_macro}_BASE + LIVE_REG_ENTRY_INDEX, entry_index);",
                f"    uint64_t data_lo = (uint64_t)mmio_read({module_macro}_BASE + LIVE_REG_READ_DATA_LO);",
                f"    uint64_t data_hi = width_bits > 32U ? (uint64_t)mmio_read({module_macro}_BASE + LIVE_REG_READ_DATA_HI) : 0ULL;",
                "    uint64_t mask = width_bits == 64U ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << width_bits) - 1ULL);",
                "    return ((data_hi << 32U) | data_lo) & mask;",
                "}",
                "",
                "static inline uint32_t live_read_status(void) {",
                f"    return mmio_read({module_macro}_BASE + LIVE_REG_STATUS);",
                "}",
                "",
                "static inline uint32_t live_read_trap_status(void) {",
                f"    return mmio_read({module_macro}_BASE + LIVE_REG_TRAP_STATUS);",
                "}",
                "",
                "static inline void live_rollback_shadow(void) {",
                f"    mmio_write({module_macro}_BASE + LIVE_REG_CONTROL, LIVE_CTRL_ROLLBACK);",
                "}",
                "",
                "static inline int live_clear_selected_traps(uint32_t trap_mask) {",
                "    if ((trap_mask & ~LIVE_TRAP_CLEAR_MASK) != 0U) {",
                "        return -1;",
                "    }",
                f"    mmio_write({module_macro}_BASE + LIVE_REG_TRAP_CLEAR, trap_mask);",
                f"    mmio_write({module_macro}_BASE + LIVE_REG_CONTROL, LIVE_CTRL_CLEAR_TRAP);",
                "    return 0;",
                "}",
                "",
                "static inline void live_clear_traps(void) {",
                "    (void)live_clear_selected_traps(LIVE_TRAP_CLEAR_MASK);",
                "}",
                "",
            ]
        )

    lines.extend(
        [
            f"static inline int32_t {module_identifier}_encode_q(float val) {{",
            f"    return (int32_t)(val * (1 << {fraction}));",
            "}",
            "",
            f"static inline void {module_identifier}_enable(void) {{",
            f"    mmio_write({module_macro}_BASE + REG_CTRL, 0x01);",
            "}",
            "",
            f"static inline void {module_identifier}_reset(void) {{",
            f"    mmio_write({module_macro}_BASE + REG_CTRL, 0x02);",
            f"    mmio_write({module_macro}_BASE + REG_CTRL, 0x01);",
            "}",
            "",
            f"static inline void {module_identifier}_set_current(float I) {{",
            f"    mmio_write({module_macro}_BASE + REG_I_T, "
            f"(uint32_t){module_identifier}_encode_q(I));",
            "}",
            "",
            f"static inline uint32_t {module_identifier}_get_spikes(void) {{",
            f"    return mmio_read({module_macro}_BASE + REG_SPIKE_COUNT);",
            "}",
        ]
    )

    for binding in param_bindings:
        register_name = binding.register_identifier.upper()
        lines.extend(
            [
                "",
                f"static inline void {module_identifier}_set_{binding.setter_identifier}(float value) {{",
                f"    mmio_write({module_macro}_BASE + REG_{register_name}, "
                f"(uint32_t){module_identifier}_encode_q(value));",
                "}",
            ]
        )

    if live_update_spec is not None:
        for bank in live_update_spec.banks:
            bank_id = _driver_identifier(bank.bank_name).upper()
            for parameter in bank.parameter_names:
                param_id = _driver_identifier(parameter).upper()
                suffix = f"{_driver_identifier(bank.bank_name)}_{_driver_identifier(parameter)}"
                lines.extend(
                    [
                        "",
                        f"static inline int {module_identifier}_update_live_{suffix}_encoded(uint64_t encoded_word) {{",
                        "    return live_update_encoded(",
                        f"        LIVE_BANK_{bank_id},",
                        f"        LIVE_{bank_id}_{param_id}_INDEX,",
                        f"        LIVE_{bank_id}_{param_id}_WIDTH_BITS,",
                        "        encoded_word",
                        "    );",
                        "}",
                        "",
                        f"static inline uint64_t {module_identifier}_read_live_{suffix}_encoded(void) {{",
                        "    return live_read_encoded(",
                        f"        LIVE_BANK_{bank_id},",
                        f"        LIVE_{bank_id}_{param_id}_INDEX,",
                        f"        LIVE_{bank_id}_{param_id}_WIDTH_BITS",
                        "    );",
                        "}",
                        "",
                        f"static inline int {module_identifier}_verify_live_{suffix}_encoded(uint64_t encoded_word) {{",
                        f"    int rc = {module_identifier}_update_live_{suffix}_encoded(encoded_word);",
                        "    if (rc != 0) {",
                        "        return rc;",
                        "    }",
                        f"    return {module_identifier}_read_live_{suffix}_encoded() == encoded_word ? 0 : -3;",
                        "}",
                    ]
                )

    lines.extend(["", f"#endif /* {guard} */", ""])

    return "\n".join(lines)
