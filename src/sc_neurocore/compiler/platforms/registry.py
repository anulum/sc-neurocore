# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

"""Pre-configured hardware profiles for every target platform.

Each profile encodes the optimal fixed-point configuration for a specific
hardware target, including DSP multiplier widths, overflow handling, and
rounding semantics.

Usage::

    from sc_neurocore.compiler.platforms import get_profile, list_profiles

    # Compile for Intel Loihi 2
    profile = get_profile("loihi2")
    verilog = neuron.to_verilog(
        module_name="sc_lif",
        data_width=profile.data_width,
        fraction=profile.fraction,
    )

    # List all available targets
    for p in list_profiles():
        print(f"{p.name:16s} {p.vendor:12s} Q{p.int_bits}.{p.fraction} ({p.data_width}-bit)")

Supported Platform Classes
--------------------------
- **Xilinx FPGA** (Spartan-6 through Versal)
- **Intel FPGA** (Cyclone V through Agilex)
- **Lattice** (ECP5, CrossLink-NX, CertusPro-NX)
- **Gowin**, **Efinix**, **Microchip/Microsemi**, **Achronix**, **QuickLogic**
- **Neuromorphic** (Loihi 2, TrueNorth, BrainScaleS-2, SpiNNaker 2, Akida, Dynap)
- **Photonic** (Lightmatter, Xanadu, iPronics, Lightelligence, Luminous)
- **In-Memory / PIM** (UPMEM, Samsung HBM-PIM, SK Hynix AiM, CXL)
- **Superconducting** (NIST SFQ, Northrop AQFP, Josephson)
- **Spintronic** (Everspin STT-MRAM, Samsung SOT-MRAM)
- **Ferroelectric** (GlobalFoundries FeFET, SK Hynix FeRAM)
- **CGRA** (Samsung, Qualcomm NPU, Cadence Xtensa)
- **3D-Stacked** (TSMC SoIC, Intel Foveros, AMD 3D V-Cache)
- **Edge MCU** (RP2040, ESP32-S3, STM32H7, nRF5340, MAX78000)
- **ASIC** (arbitrary standard-cell targets)
- **Simulation** (golden reference)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


# ── Type aliases ─────────────────────────────────────────────────────
OverflowMode = Literal["saturate", "wrap", "trap"]
RoundingMode = Literal["truncate", "nearest", "bankers", "stochastic"]


@dataclass(frozen=True)
class HardwareProfile:
    """Complete hardware configuration for a target platform.

    Attributes
    ----------
    name : str
        Short machine-readable identifier (e.g. ``"loihi2"``).
    vendor : str
        Chip vendor (e.g. ``"Intel"``, ``"Xilinx"``).
    family : str
        Product family (e.g. ``"Arria 10"``, ``"ECP5"``).
    platform_class : str
        One of ``"fpga"``, ``"neuromorphic"``, ``"asic"``, ``"simulation"``.
    data_width : int
        Total bit width for fixed-point arithmetic.
    fraction : int
        Number of fractional bits.
    signed : bool
        True for signed (two's complement), False for unsigned Q-format.
    overflow : OverflowMode
        How to handle arithmetic overflow in next-state logic.
    rounding : RoundingMode
        How to round after fixed-point multiplication truncation.
    dsp_block : str
        Name of the DSP hard macro (e.g. ``"DSP48E2"``).
    dsp_mult_a : int
        Width of the DSP A-port (multiplier input A).
    dsp_mult_b : int
        Width of the DSP B-port (multiplier input B).
    max_freq_mhz : int | None
        Typical maximum clock frequency (0 = unknown).
    notes : str
        Human-readable rationale for the configuration.
    """

    name: str
    vendor: str
    family: str
    platform_class: str  # fpga | neuromorphic | asic | simulation
    data_width: int
    fraction: int
    signed: bool = True
    overflow: OverflowMode = "saturate"
    rounding: RoundingMode = "truncate"
    dsp_block: str = ""
    dsp_mult_a: int = 0
    dsp_mult_b: int = 0
    max_freq_mhz: int | None = None
    notes: str = ""

    @property
    def int_bits(self) -> int:
        """Number of integer bits (excluding sign bit if signed)."""
        return self.data_width - self.fraction - (1 if self.signed else 0)

    @property
    def q_format_label(self) -> str:
        """Human-readable Q-format string (e.g. ``'Q9.9'`` or ``'UQ8.8'``)."""
        prefix = "Q" if self.signed else "UQ"
        return f"{prefix}{self.int_bits}.{self.fraction}"

    @property
    def max_value(self) -> float:
        """Maximum representable positive value."""
        if self.signed:
            return ((1 << (self.data_width - 1)) - 1) / (1 << self.fraction)
        return ((1 << self.data_width) - 1) / (1 << self.fraction)

    @property
    def min_value(self) -> float:
        """Minimum representable value (most negative or zero)."""
        if self.signed:
            return -(1 << (self.data_width - 1)) / (1 << self.fraction)
        return 0.0

    @property
    def resolution(self) -> float:
        """Smallest representable step."""
        return 1.0 / (1 << self.fraction)

    @classmethod
    def from_constraints(
        cls,
        name: str,
        *,
        vendor: str = "Generic",
        family: str = "Auto",
        platform_class: str = "custom",
        data_width: int | None = None,
        fraction: int | None = None,
        max_freq_mhz: int | None = None,
        overflow: OverflowMode = "saturate",
        rounding: RoundingMode = "nearest",
        min_precision_bits: int = 8,
        max_power_budget_mw: float | None = None,
        notes: str = "",
    ) -> "HardwareProfile":
        """Auto-construct an optimal profile from spec-sheet constraints.

            This is the **ultimate extensibility mechanism**: instead of manually
            defining every field, provide constraints and let SC-NeuroCore select
            the optimal fixed-point configuration.

        Parameters
        ----------
            name : str
                Unique profile identifier.
            vendor : str
                Vendor name.
            family : str
                Product family.
            platform_class : str
                Platform class identifier.
            data_width : int, optional
                Override total bit width. Auto-selects if None.
            fraction : int, optional
                Override fraction bits. Auto-selects if None.
        max_freq_mhz : int | None
                Maximum clock frequency.
            overflow : OverflowMode
                Overflow handling.
            rounding : RoundingMode
                Rounding mode.
            min_precision_bits : int
                Minimum fractional precision required.
            max_power_budget_mw : float, optional
                Power budget constraint (used for width selection).
            notes : str
                Human-readable description.

        Returns
        -------
            HardwareProfile
                Auto-constructed profile.
        """
        # Auto-select data width based on precision and power
        if data_width is None:
            if max_power_budget_mw is not None and max_power_budget_mw < 10:
                data_width = max(8, min_precision_bits)
            elif max_power_budget_mw is not None and max_power_budget_mw < 100:
                data_width = max(16, min_precision_bits * 2)
            else:
                data_width = max(16, min_precision_bits * 2)

        # Auto-select fraction: half the data width, at least min_precision
        if fraction is None:
            fraction = max(min_precision_bits, data_width // 2)
            fraction = min(fraction, data_width - 1)

        profile = cls(
            name=name,
            vendor=vendor,
            family=family,
            platform_class=platform_class,
            data_width=data_width,
            fraction=fraction,
            overflow=overflow,
            rounding=rounding,
            max_freq_mhz=max_freq_mhz,
            notes=notes or "Auto-constructed from constraints.",
        )
        # Auto-register
        _PROFILES[name.lower().replace("-", "_").replace(" ", "_")] = profile
        return profile


# ═══════════════════════════════════════════════════════════════════════
# Pre-configured profiles
# ═══════════════════════════════════════════════════════════════════════

_PROFILES: dict[str, HardwareProfile] = {}
_DISCOVERY_HOOKS: list[Callable[[], list[HardwareProfile]]] = []


def _reg(p: HardwareProfile, *, allow_override: bool = False) -> HardwareProfile:
    """Register a profile in the global registry.

    Parameters
    ----------
    p : HardwareProfile
        The profile to register, keyed by its ``name``.
    allow_override : bool, optional
        When ``False`` (the default) a name already present in the registry
        raises :class:`ValueError`. This guards the built-in profile modules
        against the silent last-wins overwrite that previously let one
        platform be registered several times with conflicting fields. Set to
        ``True`` for the user-extension paths (e.g. loading a TOML profile
        that deliberately overrides a built-in target).

    Returns
    -------
    HardwareProfile
        The registered profile (``p`` unchanged), for convenient chaining.

    Raises
    ------
    ValueError
        If ``p.name`` is already registered and ``allow_override`` is ``False``.
    """
    if not allow_override and p.name in _PROFILES:
        raise ValueError(
            f"Duplicate hardware-profile registration for '{p.name}'. "
            "Each built-in profile name must be registered exactly once; "
            "pass allow_override=True to intentionally replace a profile."
        )
    _PROFILES[p.name] = p
    return p


def get_profile(name: str) -> HardwareProfile:
    """Look up a hardware profile by name.

    Parameters
    ----------
    name : str
        Case-insensitive profile name (e.g. ``"loihi2"``, ``"artix7"``).

    Returns
    -------
    HardwareProfile
        The matching profile.

    Raises
    ------
    KeyError
        If no profile matches.
    """
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in _PROFILES:
        available = ", ".join(sorted(_PROFILES.keys()))
        raise KeyError(f"Unknown hardware profile '{name}'. Available: {available}")
    return _PROFILES[key]


def list_profiles(
    *,
    platform_class: str | None = None,
    vendor: str | None = None,
) -> list[HardwareProfile]:
    """List all registered hardware profiles, optionally filtered.

    Parameters
    ----------
    platform_class : str, optional
        Filter by class: ``"fpga"``, ``"neuromorphic"``, ``"asic"``, ``"simulation"``
        ``"accelerator"``, ``"dsp"``, ``"photonic"``, ``"in_memory"``, ``"emerging"``.
    vendor : str, optional
        Filter by vendor name (case-insensitive substring match).

    Returns
    -------
    list[HardwareProfile]
        Matching profiles, sorted by (platform_class, vendor, name).
    """
    result = list(_PROFILES.values())
    if platform_class:
        result = [p for p in result if p.platform_class == platform_class]
    if vendor:
        v_lower = vendor.lower()
        result = [p for p in result if v_lower in p.vendor.lower()]
    return sorted(result, key=lambda p: (p.platform_class, p.vendor, p.name))


def list_profile_names() -> list[str]:
    """Return all registered profile names, sorted."""
    return sorted(_PROFILES.keys())


def load_toml_profile(path: str) -> HardwareProfile:
    """Load a user-defined hardware profile from a TOML file.

    Enables users to register custom hardware targets without modifying
    the SC-NeuroCore source. TOML format::

        [profile]
        name = "my_chip"
        vendor = "My Corp"
        family = "ChipNet-1"
        platform_class = "accelerator"
        data_width = 16
        fraction = 8
        overflow = "saturate"
        rounding = "nearest"
        max_freq_mhz = 500
        dsp_block = "MAC"
        dsp_mult_a = 16
        dsp_mult_b = 16
        notes = "Custom chip description."

    Parameters
    ----------
    path : str
        Path to the TOML file.

    Returns
    -------
    HardwareProfile
        The loaded and registered profile.

    Raises
    ------
    FileNotFoundError
        If the TOML file does not exist.
    ValueError
        If required fields are missing.
    """
    from pathlib import Path

    toml_path = Path(path)
    if not toml_path.exists():
        raise FileNotFoundError(f"Profile TOML not found: {path}")

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    p = data.get("profile", data)

    required = {
        "name",
        "vendor",
        "family",
        "platform_class",
        "data_width",
        "fraction",
        "overflow",
        "rounding",
    }
    missing = required - set(p.keys())
    if missing:
        raise ValueError(f"Missing required fields in TOML profile: {missing}")

    profile = HardwareProfile(
        name=p["name"],
        vendor=p["vendor"],
        family=p["family"],
        platform_class=p["platform_class"],
        data_width=int(p["data_width"]),
        fraction=int(p["fraction"]),
        overflow=p["overflow"],
        rounding=p["rounding"],
        max_freq_mhz=int(p["max_freq_mhz"]) if "max_freq_mhz" in p else None,
        dsp_block=p.get("dsp_block", ""),
        dsp_mult_a=int(p.get("dsp_mult_a", 0)),
        dsp_mult_b=int(p.get("dsp_mult_b", 0)),
        notes=p.get("notes", "User-defined profile."),
    )
    # User-defined profiles may deliberately replace a built-in target.
    _reg(profile, allow_override=True)
    return profile


def load_toml_profiles_dir(directory: str) -> list[HardwareProfile]:
    """Load all TOML profiles from a directory.

    Scans the directory for ``*.toml`` files and loads each as a hardware
    profile. Useful for bulk-registering custom targets.

    Parameters
    ----------
    directory : str
        Path to the directory containing TOML profile files.

    Returns
    -------
    list[HardwareProfile]
        All loaded profiles.
    """
    from pathlib import Path

    profiles: list[HardwareProfile] = []
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return profiles
    for toml_file in sorted(dir_path.glob("*.toml")):
        profiles.append(load_toml_profile(str(toml_file)))
    return profiles


def load_profiles_from_toml(path: str) -> list[str]:
    """Load custom hardware profiles from a TOML file.

    Allows users and vendors to define profiles without modifying
    SC-NeuroCore source code. This is the profile-extension path.

    TOML format::

        [[profile]]
        name = "my_custom_chip"
        vendor = "MyVendor"
        family = "CustomFamily"
        platform_class = "custom"
        data_width = 16
        fraction = 8
        overflow = "saturate"
        rounding = "nearest"

    Parameters
    ----------
    path : str
        Path to TOML file.

    Returns
    -------
    list[str]
        Names of loaded profiles.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    loaded = []
    for entry in data.get("profile", []):
        p = HardwareProfile(
            name=entry["name"],
            vendor=entry.get("vendor", "Custom"),
            family=entry.get("family", "Custom"),
            platform_class=entry.get("platform_class", "custom"),
            data_width=entry.get("data_width", 16),
            fraction=entry.get("fraction", 8),
            overflow=entry.get("overflow", "saturate"),
            rounding=entry.get("rounding", "nearest"),
            dsp_block=entry.get("dsp_block"),
            dsp_mult_a=entry.get("dsp_mult_a"),
            dsp_mult_b=entry.get("dsp_mult_b"),
            max_freq_mhz=entry.get("max_freq_mhz"),
            notes=entry.get("notes", ""),
        )
        _PROFILES[p.name] = p
        loaded.append(p.name)

    return loaded


def register_platform_hook(hook_fn: Callable[[], list[HardwareProfile]]) -> None:
    """Register a third-party platform discovery function.

    The hook function should return a list of HardwareProfile instances
    when called with no arguments. Profiles are registered at runtime.

    Parameters
    ----------
    hook_fn : callable
        Function returning list[HardwareProfile].
    """
    _DISCOVERY_HOOKS.append(hook_fn)


def discover_platforms() -> list[str]:
    """Execute all registered discovery hooks.

    Returns
    -------
    list[str]
        Names of newly discovered profiles.
    """
    discovered = []
    for hook in _DISCOVERY_HOOKS:
        profiles = hook()
        for p in profiles:
            if p.name not in _PROFILES:
                _PROFILES[p.name] = p
                discovered.append(p.name)
    return discovered
