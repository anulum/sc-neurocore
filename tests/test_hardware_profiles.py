# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware profile and advanced compiler feature tests

"""Tests for hardware profiles, overflow modes, rounding modes, and unsigned Q-format.

Covers:
- HardwareProfile registry (30+ profiles)
- Compilation with --target (every FPGA/neuromorphic/ASIC platform)
- Overflow modes: saturate, wrap, trap
- Rounding modes: truncate, nearest, bankers, stochastic
- Unsigned Q-format support
- Cross-platform co-simulation where iverilog is available
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.equation_compiler import Q88, compile_to_verilog
from sc_neurocore.compiler.hardware_profiles import (
    HardwareProfile,
    get_profile,
    list_profiles,
    list_profile_names,
)
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

HAS_IVERILOG = shutil.which("iverilog") is not None


# ═══════════════════════════════════════════════════════════════════════
# Hardware Profile Registry Tests
# ═══════════════════════════════════════════════════════════════════════

class TestHardwareProfileRegistry:
    """Test that the profile registry is complete and consistent."""

    def test_at_least_30_profiles(self) -> None:
        """We should have 30+ pre-configured profiles."""
        profiles = list_profiles()
        assert len(profiles) >= 30, f"Only {len(profiles)} profiles registered"

    def test_all_platform_classes_present(self) -> None:
        """Every platform class must have at least one profile."""
        profiles = list_profiles()
        classes = {p.platform_class for p in profiles}
        for expected in ("fpga", "neuromorphic", "asic", "simulation"):
            assert expected in classes, f"Missing platform class: {expected}"

    def test_fpga_vendor_coverage(self) -> None:
        """Major FPGA vendors must be covered."""
        profiles = list_profiles(platform_class="fpga")
        vendors = {p.vendor for p in profiles}
        for v in ("Xilinx", "Intel", "Lattice", "Gowin", "Efinix", "Microchip"):
            assert v in vendors, f"Missing FPGA vendor: {v}"

    def test_neuromorphic_chip_coverage(self) -> None:
        """Major neuromorphic chips must be covered."""
        names = list_profile_names()
        for chip in ("loihi2", "truenorth", "spinnaker2", "akida"):
            assert chip in names, f"Missing neuromorphic chip: {chip}"

    def test_get_profile_case_insensitive(self) -> None:
        """Profile lookup should be case-insensitive."""
        p = get_profile("Loihi2")
        assert p.name == "loihi2"

    def test_get_profile_unknown_raises(self) -> None:
        """Unknown profile names should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown hardware profile"):
            get_profile("nonexistent_chip_xyz")

    def test_filter_by_platform_class(self) -> None:
        """Filtering by platform_class works."""
        fpga = list_profiles(platform_class="fpga")
        neuro = list_profiles(platform_class="neuromorphic")
        assert all(p.platform_class == "fpga" for p in fpga)
        assert all(p.platform_class == "neuromorphic" for p in neuro)

    def test_filter_by_vendor(self) -> None:
        """Filtering by vendor works."""
        xilinx = list_profiles(vendor="Xilinx")
        assert len(xilinx) >= 4
        assert all("Xilinx" in p.vendor for p in xilinx)

    @pytest.mark.parametrize("name", list_profile_names())
    def test_profile_properties(self, name: str) -> None:
        """Every profile must have valid Q-format properties."""
        p = get_profile(name)
        assert p.data_width > 0
        assert 0 < p.fraction < p.data_width
        assert p.resolution > 0
        assert p.max_value > p.min_value
        assert p.q_format_label  # non-empty string


# ═══════════════════════════════════════════════════════════════════════
# Q88 Dataclass Extension Tests
# ═══════════════════════════════════════════════════════════════════════

class TestQ88Extensions:
    """Test the signed/unsigned, overflow, and rounding fields on Q88."""

    def test_signed_defaults(self) -> None:
        """Default Q88 is signed."""
        q = Q88()
        assert q.signed is True
        assert q.overflow == "saturate"
        assert q.rounding == "truncate"

    def test_unsigned_range(self) -> None:
        """Unsigned Q8.8 should have range [0, 255.996]."""
        q = Q88(data_width=16, fraction=8, signed=False)
        assert q.min_value == 0.0
        assert q.max_value == pytest.approx(255.99609375)
        assert q.integer_bits == 8  # no sign bit consumed

    def test_signed_vs_unsigned_integer_bits(self) -> None:
        """Unsigned format should have 1 more integer bit."""
        qs = Q88(data_width=16, fraction=8, signed=True)
        qu = Q88(data_width=16, fraction=8, signed=False)
        assert qu.integer_bits == qs.integer_bits + 1

    def test_overflow_modes_accepted(self) -> None:
        """All three overflow modes are valid."""
        for mode in ("saturate", "wrap", "trap"):
            q = Q88(overflow=mode)
            assert q.overflow == mode

    def test_rounding_modes_accepted(self) -> None:
        """All four rounding modes are valid."""
        for mode in ("truncate", "nearest", "bankers", "stochastic"):
            q = Q88(rounding=mode)
            assert q.rounding == mode


# ═══════════════════════════════════════════════════════════════════════
# Compilation Tests: Every Profile Compiles
# ═══════════════════════════════════════════════════════════════════════

# Profiles where LIF params fit (v_rest=-65 needs integer range >= 65)
_WIDE_RANGE_PROFILES = [
    name for name in list_profile_names()
    if get_profile(name).min_value <= -65
]

_NARROW_RANGE_PROFILES = [
    name for name in list_profile_names()
    if get_profile(name).min_value > -65
]


class TestCompilationAllProfiles:
    """Ensure every hardware profile compiles without errors."""

    @pytest.mark.parametrize("profile_name", _WIDE_RANGE_PROFILES)
    def test_compile_lif_wide_range(self, profile_name: str) -> None:
        """LIF compiles for wide-range profiles without error."""
        p = get_profile(profile_name)
        neuron = UniversalNeuron.from_schema("lif")
        eq = neuron.to_equation_neuron()
        verilog = compile_to_verilog(
            eq, module_name=f"sc_lif_{profile_name}",
            data_width=p.data_width, fraction=p.fraction,
            overflow=p.overflow, rounding=p.rounding,
        )
        assert "module sc_lif_" in verilog
        assert f"{p.data_width}-bit" in verilog or f"[{p.data_width - 1}:0]" in verilog

    @pytest.mark.parametrize("profile_name", _NARROW_RANGE_PROFILES)
    def test_compile_narrow_range_no_crash(self, profile_name: str) -> None:
        """Narrow-range profiles compile without crashing (may not be accurate)."""
        p = get_profile(profile_name)
        neuron = UniversalNeuron.from_schema("resonate_fire")
        eq = neuron.to_equation_neuron()
        verilog = compile_to_verilog(
            eq, module_name=f"sc_rf_{profile_name}",
            data_width=p.data_width, fraction=p.fraction,
            overflow=p.overflow, rounding=p.rounding,
        )
        assert "module sc_rf_" in verilog


# ═══════════════════════════════════════════════════════════════════════
# Overflow Mode Tests
# ═══════════════════════════════════════════════════════════════════════

class TestOverflowModes:
    """Verify Verilog output contains correct overflow handling logic."""

    def _compile_lif(self, overflow: str) -> str:
        """Helper: compile LIF with given overflow mode."""
        neuron = UniversalNeuron.from_schema("lif")
        eq = neuron.to_equation_neuron()
        return compile_to_verilog(
            eq, module_name="sc_lif_ovf",
            data_width=16, fraction=8,
            overflow=overflow,
        )

    def test_saturate_has_clamp(self) -> None:
        """Saturate mode must emit ternary clamp logic."""
        v = self._compile_lif("saturate")
        assert "?" in v  # ternary operator for clamp
        assert "32767" in v or "sd32767" in v  # max value

    def test_wrap_no_clamp(self) -> None:
        """Wrap mode must NOT emit clamp logic."""
        v = self._compile_lif("wrap")
        # Should have simple bit-select, no ternary clamp
        assert "$fatal" not in v
        # The key difference: wrap uses direct bit-select
        assert "v_raw[15:0]" in v

    def test_trap_has_fatal(self) -> None:
        """Trap mode must emit $fatal assertion."""
        v = self._compile_lif("trap")
        assert "$fatal" in v
        assert "OVERFLOW TRAP" in v
        assert "synthesis translate_off" in v
        assert "synthesis translate_on" in v


# ═══════════════════════════════════════════════════════════════════════
# Rounding Mode Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRoundingModes:
    """Verify Verilog output contains correct rounding logic."""

    def _compile_lif(self, rounding: str) -> str:
        """Helper: compile LIF with given rounding mode."""
        neuron = UniversalNeuron.from_schema("lif")
        eq = neuron.to_equation_neuron()
        return compile_to_verilog(
            eq, module_name="sc_lif_rnd",
            data_width=16, fraction=8,
            rounding=rounding,
        )

    def test_truncate_simple_shift(self) -> None:
        """Truncate mode: simple arithmetic right shift."""
        v = self._compile_lif("truncate")
        assert ">>> 8" in v
        # Should NOT have rounding bias wires
        assert "_rnd_half" not in v

    def test_nearest_has_half_bias(self) -> None:
        """Nearest mode: add 0.5 LSB before shift."""
        v = self._compile_lif("nearest")
        assert "_rnd_half" in v
        assert "128" in v  # 1 << (8-1) = 128

    def test_bankers_has_guard(self) -> None:
        """Banker's rounding: has guard bit logic."""
        v = self._compile_lif("bankers")
        assert "_rnd_guard" in v
        assert "_rnd_biased" in v

    def test_stochastic_has_lfsr(self) -> None:
        """Stochastic rounding: references LFSR."""
        v = self._compile_lif("stochastic")
        assert "_rnd_stoch" in v
        assert "_lfsr" in v


# ═══════════════════════════════════════════════════════════════════════
# Co-Simulation Tests: Overflow & Rounding (iverilog required)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestCoSimOverflowRounding:
    """Verify overflow and rounding modes in actual Verilog simulation."""

    def _run_cosim(
        self, model: str, dw: int, frac: int,
        overflow: str, rounding: str, n_steps: int = 100, current: float = 50.0,
    ) -> int:
        """Compile + simulate, return spike count."""
        from sc_neurocore.compiler.equation_compiler import generate_testbench

        neuron = UniversalNeuron.from_schema(model)
        eq = neuron.to_equation_neuron()
        mod = f"sc_{model}_ovf"

        verilog = compile_to_verilog(
            eq, module_name=mod,
            data_width=dw, fraction=frac,
            overflow=overflow, rounding=rounding,
        )
        tb = generate_testbench(
            eq, module_name=mod,
            n_steps=n_steps, input_current=current,
            data_width=dw, fraction=frac,
        )

        with tempfile.TemporaryDirectory() as d:
            Path(f"{d}/{mod}.v").write_text(verilog)
            Path(f"{d}/tb.v").write_text(tb)
            r = subprocess.run(
                ["iverilog", "-g2012", "-o", f"{d}/tb",
                 f"{d}/{mod}.v", f"{d}/tb.v"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                raise RuntimeError(f"iverilog failed:\n{r.stderr}")
            r = subprocess.run(
                ["vvp", f"{d}/tb"],
                capture_output=True, text=True, timeout=30,
            )
            m = re.search(r"(\d+) spikes", r.stdout)
            return int(m.group(1)) if m else -1

    def test_saturate_vs_wrap_lif(self) -> None:
        """Saturate and wrap should produce same result for LIF (no overflow)."""
        sat = self._run_cosim("lif", 16, 8, "saturate", "truncate")
        wrap = self._run_cosim("lif", 16, 8, "wrap", "truncate")
        # Both should produce the same count (LIF doesn't overflow in Q8.8)
        assert sat == wrap
        assert sat > 0

    def test_nearest_rounding_lif(self) -> None:
        """Nearest rounding should also produce spikes for LIF."""
        spikes = self._run_cosim("lif", 16, 8, "saturate", "nearest")
        assert spikes > 0

    @pytest.mark.parametrize("profile_name", [
        "artix7", "ecp5", "loihi2", "versal", "ice40",
    ])
    def test_profile_cosim(self, profile_name: str) -> None:
        """Key profiles should compile and simulate successfully."""
        p = get_profile(profile_name)
        if p.min_value > -65:
            pytest.skip(f"{profile_name}: LIF params overflow range")
        spikes = self._run_cosim(
            "lif", p.data_width, p.fraction,
            p.overflow, p.rounding,
        )
        assert spikes > 0, f"{profile_name}: no spikes in co-simulation"
