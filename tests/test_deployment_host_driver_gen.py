# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHostDriverGen from former test_deployment.py

"""Focused suite: TestHostDriverGen from former test_deployment.py."""

from __future__ import annotations

from tests.deployment_support import *  # noqa: F403

class TestHostDriverGen:
    """Test Python and C driver generation."""

    def test_python_driver(self) -> None:
        """Python driver should be a valid class."""
        drv = generate_host_driver("sc_lif", LIF_PARAMS, language="python")
        assert "class ScLifDriver:" in drv
        assert "def enable(self)" in drv
        assert "def set_current(self" in drv
        assert "def get_spike_count(self" in drv

    def test_python_param_setters(self) -> None:
        """Should generate setter for each parameter."""
        drv = generate_host_driver("sc_lif", LIF_PARAMS, language="python")
        assert "def set_v_rest(self" in drv
        assert "def set_v_thresh(self" in drv
        assert "def set_tau_m(self" in drv

    def test_python_register_map(self) -> None:
        """Should include register offsets."""
        drv = generate_host_driver("sc_lif", LIF_PARAMS, language="python")
        assert "REG_CTRL" in drv
        assert "0x00" in drv
        assert "REG_SPIKE_COUNT" in drv

    def test_python_q_encoding(self) -> None:
        """Should include Q-format encode/decode."""
        drv = generate_host_driver("sc_lif", LIF_PARAMS, language="python")
        assert "encode_q" in drv
        assert "decode_q" in drv

    def test_c_driver(self) -> None:
        """C driver should be a valid header."""
        drv = generate_host_driver("sc_lif", LIF_PARAMS, language="c")
        assert "#ifndef SC_LIF_DRIVER_H" in drv
        assert "#define SC_LIF_DRIVER_H" in drv
        assert "mmio_write" in drv
        assert "mmio_read" in drv

    def test_c_encode(self) -> None:
        """C driver should have Q-format encoding."""
        drv = generate_host_driver("sc_lif", LIF_PARAMS, language="c")
        assert "encode_q" in drv

    def test_c_functions(self) -> None:
        """C driver should have enable/reset/set_current."""
        drv = generate_host_driver("sc_lif", LIF_PARAMS, language="c")
        assert "sc_lif_enable" in drv
        assert "sc_lif_reset" in drv
        assert "sc_lif_set_current" in drv
        assert "sc_lif_get_spikes" in drv

    def test_custom_base_address(self) -> None:
        """Base address should be configurable."""
        drv = generate_host_driver(
            "sc_lif",
            LIF_PARAMS,
            language="python",
            base_address=0x8000_0000,
        )
        assert "80000000" in drv

    def test_invalid_language(self) -> None:
        """Should raise on invalid language."""
        with pytest.raises(ValueError, match="Unsupported language"):
            generate_host_driver("sc_lif", LIF_PARAMS, language="rust")  # type: ignore[arg-type]

    def test_python_driver_sanitises_module_and_parameter_identifiers(self) -> None:
        """Generated Python drivers should be valid for unsafe source names."""
        source = generate_host_driver(
            "12 lif/core",
            {"tau-m": 16, "P v thresh": 16},
            language="python",
        )
        namespace: dict[str, object] = {}
        exec(source, namespace)

        driver_factory = cast(_GeneratedUnsafeHostDriverFactory, namespace["P12LifCoreDriver"])
        writes: list[tuple[int, int]] = []
        driver = driver_factory(
            lambda _address: 0,
            lambda address, value: writes.append((address, value)),
        )

        driver.set_tau_m(1.25)
        driver.set_v_thresh(2.0)

        assert "REG_TAU_M" in source
        assert "REG_P_V_THRESH" in source
        assert not any(fragment in source for fragment in ("tau-m", "P v thresh"))
        assert writes == [(0x4000_000C, 320), (0x4000_0010, 512)]

    def test_c_driver_sanitises_module_and_parameter_identifiers(self) -> None:
        """Generated C drivers should sanitize guards, macros, and functions."""
        source = generate_host_driver(
            "12 lif/core",
            {"tau-m": 16, "P v thresh": 16},
            language="c",
        )

        assert "#ifndef P_12_LIF_CORE_DRIVER_H" in source
        assert "#define P_12_LIF_CORE_BASE" in source
        assert "#define REG_TAU_M" in source
        assert "#define REG_P_V_THRESH" in source
        assert "static inline void p_12_lif_core_set_tau_m" in source
        assert "static inline void p_12_lif_core_set_v_thresh" in source
        assert "tau-m" not in source

    def test_host_driver_rejects_empty_generated_module_identifier(self) -> None:
        """Generated drivers should fail closed when module names sanitize empty."""
        with pytest.raises(ValueError, match="module_name"):
            generate_host_driver("!!!", {"tau": 16}, language="python")

    def test_host_driver_rejects_colliding_parameter_identifiers(self) -> None:
        """Generated drivers should fail closed on sanitized parameter collisions."""
        with pytest.raises(ValueError, match="parameter identifier collision"):
            generate_host_driver("sc_lif", {"tau-m": 16, "tau m": 16}, language="c")

    def test_host_driver_rejects_colliding_parameter_setters(self) -> None:
        """Generated drivers should reject duplicate setters after P-prefix folding."""
        with pytest.raises(ValueError, match="parameter identifier collision"):
            generate_host_driver("sc_lif", {"P v": 16, "v": 16}, language="python")

    def test_python_live_control_driver_zeroes_high_word_and_verifies_readback(self) -> None:
        """Generated Python driver should use the full CRC/readback live-control contract."""
        spec = MMIOUpdateSpec(
            bus_protocol="axi4_lite",
            control_base_address_bytes=0x100,
            banks=(
                ParameterBankSpec(
                    bank_name="weights",
                    start_address_bytes=0x2000,
                    parameter_count=1,
                    parameter_names=("w0",),
                    q_format="Q8.8",
                ),
            ),
        )
        source = generate_host_driver(
            "sc_live",
            {},
            language="python",
            base_address=0x8000_0000,
            live_update_spec=spec,
        )
        namespace: dict[str, object] = {}
        exec(source, namespace)
        driver_factory = cast(_GeneratedLiveHostDriverFactory, namespace["ScLiveDriver"])
        writes: list[tuple[int, int]] = []

        def read_fn(address: int) -> int:
            if address == 0x8000_0104:
                return 0
            if address == 0x8000_0124:
                return 0x1234
            return 0

        def write_fn(address: int, value: int) -> None:
            writes.append((address, value))

        driver = driver_factory(read_fn, write_fn)

        assert driver.verify_live_weights_w0_encoded(0x1234) is True
        assert (0x8000_0114, 0) in writes
        assert writes[:7] == [
            (0x8000_0108, 0),
            (0x8000_010C, 0),
            (0x8000_0110, 0x1234),
            (0x8000_0114, 0),
            (0x8000_0120, spec.update_checksum("weights", "w0", 0x1234)),
            (0x8000_0100, 1),
            (0x8000_0100, 2),
        ]
        assert writes[-2:] == [(0x8000_0108, 0), (0x8000_010C, 0)]

    def test_python_live_control_driver_raises_on_trap_status(self) -> None:
        """Generated Python driver should not hide hardware trap telemetry."""
        spec = MMIOUpdateSpec(
            bus_protocol="axi4_lite",
            control_base_address_bytes=0x100,
            banks=(
                ParameterBankSpec(
                    bank_name="weights",
                    start_address_bytes=0x2000,
                    parameter_count=1,
                    parameter_names=("w0",),
                    q_format="Q8.8",
                ),
            ),
        )
        source = generate_host_driver("sc_live", {}, language="python", live_update_spec=spec)
        namespace: dict[str, object] = {}
        exec(source, namespace)
        driver_factory = cast(_GeneratedLiveHostDriverFactory, namespace["ScLiveDriver"])
        driver = driver_factory(
            lambda _address: spec.status_bits["trap_latched"],
            lambda _address, _value: None,
        )

        with pytest.raises(RuntimeError, match="hardware trap"):
            driver.update_live_weights_w0_encoded(0x1234)

    def test_python_live_control_driver_exposes_rollback_and_trap_reads(self) -> None:
        """Generated Python driver should expose the live-control recovery handshake."""
        spec = MMIOUpdateSpec(
            bus_protocol="axi4_lite",
            control_base_address_bytes=0x100,
            banks=(
                ParameterBankSpec(
                    bank_name="weights",
                    start_address_bytes=0x2000,
                    parameter_count=1,
                    parameter_names=("w0",),
                    q_format="Q8.8",
                ),
            ),
        )
        source = generate_host_driver(
            "sc_live",
            {},
            language="python",
            base_address=0x8000_0000,
            live_update_spec=spec,
        )
        namespace: dict[str, object] = {}
        exec(source, namespace)
        driver_factory = cast(_GeneratedLiveHostDriverFactory, namespace["ScLiveDriver"])
        writes: list[tuple[int, int]] = []

        def read_fn(address: int) -> int:
            if address == 0x8000_0104:
                return 0xA5
            if address == 0x8000_0118:
                return 0x5A
            raise AssertionError(f"unexpected read address 0x{address:08X}")

        def write_fn(address: int, value: int) -> None:
            writes.append((address, value))

        driver = driver_factory(read_fn, write_fn)

        assert driver.read_live_status() == 0xA5
        assert driver.read_live_trap_status() == 0x5A

        driver.rollback_live_shadow()
        assert writes == [(0x8000_0100, spec.control_bits["rollback"])]

        driver.clear_selected_live_traps(0x21)
        assert writes[-2:] == [
            (0x8000_011C, 0x21),
            (0x8000_0100, spec.control_bits["clear_trap"]),
        ]

        with pytest.raises(ValueError, match="trap_mask"):
            driver.clear_selected_live_traps(True)
        with pytest.raises(ValueError, match="live-control trap bits"):
            driver.clear_selected_live_traps(spec.trap_clear_mask + 1)

        driver.clear_live_traps()
        assert writes[-2:] == [
            (0x8000_011C, spec.trap_clear_mask),
            (0x8000_0100, spec.control_bits["clear_trap"]),
        ]

    def test_c_live_control_driver_emits_crc_update_and_readback_helpers(self) -> None:
        """Generated C driver should expose deterministic live-control helpers."""
        spec = MMIOUpdateSpec(
            bus_protocol="pcie",
            control_base_address_bytes=0x100,
            banks=(
                ParameterBankSpec(
                    bank_name="bfp_weights",
                    start_address_bytes=0x2000,
                    parameter_count=1,
                    parameter_names=("w0",),
                    precision_mode="bfp",
                    bfp_exponent_bits=12,
                    bfp_mantissa_bits=36,
                ),
            ),
        )

        drv = generate_host_driver("sc_live", {}, language="c", live_update_spec=spec)

        assert "static inline uint32_t live_update_crc32" in drv
        assert "mmio_write(SC_LIVE_BASE + LIVE_REG_WRITE_DATA_HI, data_hi);" in drv
        assert "SC_LIVE_BASE + LIVE_REG_READ_DATA_HI" in drv
        assert "LIVE_CTRL_ROLLBACK" in drv
        assert "LIVE_TRAP_CLEAR_MASK" in drv
        assert "static inline uint32_t live_read_status" in drv
        assert "static inline uint32_t live_read_trap_status" in drv
        assert "static inline void live_rollback_shadow" in drv
        assert "static inline int live_clear_selected_traps" in drv
        assert "sc_live_update_live_bfp_weights_w0_encoded" in drv
        assert "sc_live_verify_live_bfp_weights_w0_encoded" in drv

    def test_c_live_control_driver_compiles_with_readback_consumer(self, tmp_path: Path) -> None:
        """Generated C live-control helpers should compile in a real consumer."""
        cc = shutil.which("cc") or shutil.which("gcc")
        if cc is None:
            raise AssertionError(
                "a C compiler is required for generated live-control driver checks"
            )
        spec = MMIOUpdateSpec(
            bus_protocol="axi4_lite",
            control_base_address_bytes=0x100,
            banks=(
                ParameterBankSpec(
                    bank_name="weights",
                    start_address_bytes=0x2000,
                    parameter_count=1,
                    parameter_names=("w0",),
                    q_format="Q8.8",
                ),
            ),
        )
        header = generate_host_driver("sc_live", {}, language="c", live_update_spec=spec)
        header_path = tmp_path / "sc_live_driver.h"
        source_path = tmp_path / "driver_consumer.c"
        object_path = tmp_path / "driver_consumer.o"
        header_path.write_text(header, encoding="utf-8")
        source_path.write_text(
            """
#include <stdint.h>
#include "sc_live_driver.h"

static uint32_t observed_addr;
static uint32_t observed_val;

void mmio_write(uint32_t addr, uint32_t val) {
    observed_addr = addr;
    observed_val = val;
}

uint32_t mmio_read(uint32_t addr) {
    if (addr == SC_LIVE_BASE + LIVE_REG_STATUS) {
        return 0U;
    }
    if (addr == SC_LIVE_BASE + LIVE_REG_TRAP_STATUS) {
        return 0U;
    }
    if (addr == SC_LIVE_BASE + LIVE_REG_READ_DATA_LO) {
        return 0x00001234U;
    }
    return 0U;
}

int main(void) {
    int rc = sc_live_verify_live_weights_w0_encoded(0x1234ULL);
    live_rollback_shadow();
    int clear_rc = live_clear_selected_traps(0x1U);
    live_clear_traps();
    uint32_t status = live_read_status();
    uint32_t trap_status = live_read_trap_status();
    return rc == 0 && clear_rc == 0 && status == 0U && trap_status == 0U && observed_addr != 0U && observed_val != 0xFFFFFFFFU ? 0 : 1;
}
""",
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                cc,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-c",
                str(source_path),
                "-o",
                str(object_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
