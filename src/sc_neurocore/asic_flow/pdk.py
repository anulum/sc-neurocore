# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC process-design-kit configuration and validation

"""Resolve and validate process-design-kit inputs for ASIC deck generation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class PDKType(Enum):
    """Process-design-kit families supported by the deck templates."""

    SKY130 = "sky130"
    GF180MCU = "gf180mcu"
    TSMC28 = "tsmc28"
    INTEL16 = "intel16"
    CUSTOM = "custom"


@dataclass
class PDKConfig:
    """Process Design Kit configuration."""

    pdk_type: PDKType = PDKType.SKY130
    liberty_file: str = ""
    lef_file: str = ""
    tech_lef: str = ""
    cell_prefix: str = "sky130_fd_sc_hd__"
    clock_period_ns: float = 10.0
    voltage_v: float = 1.8
    temperature_c: float = 25.0
    corner: str = "tt"
    metal_layers: int = 5
    min_feature_nm: int = 130

    @classmethod
    def from_pdk_type(cls, pdk: PDKType) -> PDKConfig:
        """Construct the maintained preset for a process family.

        Parameters
        ----------
        pdk:
            Process family whose nominal files, voltage, and geometry are
            required.

        Returns
        -------
        PDKConfig
            Configuration containing deterministic ``$PDK_ROOT`` templates.
        """
        presets: Dict[PDKType, Dict[str, Any]] = {
            PDKType.SKY130: dict(
                liberty_file="$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib",
                lef_file="$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef",
                tech_lef="$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__nom.tlef",
                cell_prefix="sky130_fd_sc_hd__",
                clock_period_ns=10.0,
                voltage_v=1.8,
                metal_layers=5,
                min_feature_nm=130,
            ),
            PDKType.GF180MCU: dict(
                liberty_file="$PDK_ROOT/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lib/gf180mcu_fd_sc_mcu7t5v0__tt_025C_3v30.lib",
                lef_file="$PDK_ROOT/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/lef/gf180mcu_fd_sc_mcu7t5v0.lef",
                tech_lef="$PDK_ROOT/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/techlef/gf180mcu_fd_sc_mcu7t5v0__nom.tlef",
                cell_prefix="gf180mcu_fd_sc_mcu7t5v0__",
                clock_period_ns=15.0,
                voltage_v=3.3,
                metal_layers=6,
                min_feature_nm=180,
            ),
            PDKType.TSMC28: dict(
                liberty_file="$PDK_ROOT/tsmc28/tcbn28hpcplusbwp7t30p140_110a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp7t30p140ssgnp0p81v125c.lib",
                lef_file="$PDK_ROOT/tsmc28/lef/tcbn28hpcplusbwp7t30p140.lef",
                tech_lef="$PDK_ROOT/tsmc28/lef/HiPe_M10.tlef",
                cell_prefix="TSMC_",
                clock_period_ns=2.0,
                voltage_v=0.9,
                metal_layers=10,
                min_feature_nm=28,
            ),
            PDKType.INTEL16: dict(
                liberty_file="$PDK_ROOT/intel16/lib/intel16_sc.lib",
                lef_file="$PDK_ROOT/intel16/lef/intel16_sc.lef",
                tech_lef="$PDK_ROOT/intel16/lef/intel16.tlef",
                cell_prefix="INTEL16_",
                clock_period_ns=1.5,
                voltage_v=0.8,
                metal_layers=12,
                min_feature_nm=16,
            ),
            PDKType.CUSTOM: dict(
                liberty_file="",
                lef_file="",
                tech_lef="",
                cell_prefix="",
                clock_period_ns=10.0,
                voltage_v=1.8,
                metal_layers=5,
                min_feature_nm=130,
            ),
        }
        return cls(pdk_type=pdk, **presets[pdk])

    @property
    def is_open_source(self) -> bool:
        """Return whether the process has a maintained open-source file map."""
        return self.pdk_type in (PDKType.SKY130, PDKType.GF180MCU)

    def with_pdk_root(self, pdk_root: str) -> PDKConfig:
        """Return a copy with ``$PDK_ROOT`` variables bound to ``pdk_root``."""
        root = str(Path(pdk_root).expanduser())
        return PDKConfig(
            pdk_type=self.pdk_type,
            liberty_file=self.liberty_file.replace("$PDK_ROOT", root),
            lef_file=self.lef_file.replace("$PDK_ROOT", root),
            tech_lef=self.tech_lef.replace("$PDK_ROOT", root),
            cell_prefix=self.cell_prefix,
            clock_period_ns=self.clock_period_ns,
            voltage_v=self.voltage_v,
            temperature_c=self.temperature_c,
            corner=self.corner,
            metal_layers=self.metal_layers,
            min_feature_nm=self.min_feature_nm,
        )


@dataclass(frozen=True)
class ResolvedPDKFiles:
    """Resolved file paths required by the open-source ASIC flow."""

    liberty_file: str
    lef_file: str
    tech_lef: str
    setup_tcl: str = ""
    drc_deck: str = ""
    lvs_setup: str = ""

    def required_paths(self) -> Dict[str, str]:
        """Return the Liberty, cell-LEF, and technology-LEF paths."""
        return {
            "liberty_file": self.liberty_file,
            "lef_file": self.lef_file,
            "tech_lef": self.tech_lef,
        }

    def optional_paths(self) -> Dict[str, str]:
        """Return optional setup, DRC-deck, and LVS-setup paths."""
        return {
            "setup_tcl": self.setup_tcl,
            "drc_deck": self.drc_deck,
            "lvs_setup": self.lvs_setup,
        }


@dataclass(frozen=True)
class PDKResolution:
    """Outcome of resolving a PDK against the local filesystem."""

    pdk: PDKConfig
    files: ResolvedPDKFiles
    missing_required: Tuple[str, ...] = ()
    missing_optional: Tuple[str, ...] = ()

    @property
    def usable_for_synthesis(self) -> bool:
        """Return whether every synthesis-required PDK file was found."""
        return not self.missing_required

    @property
    def usable_for_signoff(self) -> bool:
        """Return whether required and optional signoff files were found."""
        return self.usable_for_synthesis and not self.missing_optional


class OpenSourcePDKResolver:
    """Resolve Sky130/GF180 file locations without requiring OpenLane at import time."""

    @staticmethod
    def resolve(
        pdk: PDKConfig,
        pdk_root: Optional[str] = None,
        require_existing: bool = False,
    ) -> PDKResolution:
        """Bind ``$PDK_ROOT`` and report missing PDK artefacts.

        Parameters
        ----------
        pdk:
            PDK preset or custom configuration.
        pdk_root:
            Explicit PDK root. If absent, ``PDK_ROOT`` then ``PDKPATH`` are used.
        require_existing:
            When true, missing required files are reported as blockers. When false,
            paths are still resolved so generated flow decks are deterministic.
        """
        root = pdk_root or os.environ.get("PDK_ROOT") or os.environ.get("PDKPATH") or "$PDK_ROOT"
        resolved_pdk = pdk.with_pdk_root(root)
        files = OpenSourcePDKResolver._file_manifest(resolved_pdk)

        missing_required: Tuple[str, ...] = ()
        missing_optional: Tuple[str, ...] = ()
        if require_existing:
            missing_required = tuple(
                name for name, path in files.required_paths().items() if not Path(path).exists()
            )
            missing_optional = tuple(
                name
                for name, path in files.optional_paths().items()
                if path and not Path(path).exists()
            )

        return PDKResolution(resolved_pdk, files, missing_required, missing_optional)

    @staticmethod
    def _file_manifest(pdk: PDKConfig) -> ResolvedPDKFiles:
        if pdk.pdk_type == PDKType.SKY130:
            root = OpenSourcePDKResolver._pdk_root_from_path(pdk.liberty_file, "sky130A")
            return ResolvedPDKFiles(
                liberty_file=pdk.liberty_file,
                lef_file=pdk.lef_file,
                tech_lef=pdk.tech_lef,
                setup_tcl=f"{root}/sky130A/libs.tech/netgen/sky130A_setup.tcl",
                drc_deck=f"{root}/sky130A/libs.tech/klayout/drc/sky130.lydrc",
                lvs_setup=f"{root}/sky130A/libs.tech/netgen/sky130A_setup.tcl",
            )
        if pdk.pdk_type == PDKType.GF180MCU:
            root = OpenSourcePDKResolver._pdk_root_from_path(pdk.liberty_file, "gf180mcuD")
            return ResolvedPDKFiles(
                liberty_file=pdk.liberty_file,
                lef_file=pdk.lef_file,
                tech_lef=pdk.tech_lef,
                setup_tcl=f"{root}/gf180mcuD/libs.tech/netgen/gf180mcuD_setup.tcl",
                drc_deck=f"{root}/gf180mcuD/libs.tech/klayout/drc/gf180mcu.drc",
                lvs_setup=f"{root}/gf180mcuD/libs.tech/netgen/gf180mcuD_setup.tcl",
            )
        return ResolvedPDKFiles(
            liberty_file=pdk.liberty_file,
            lef_file=pdk.lef_file,
            tech_lef=pdk.tech_lef,
        )

    @staticmethod
    def _pdk_root_from_path(path: str, marker: str) -> str:
        before, separator, _after = path.partition(f"/{marker}/")
        return before if separator else "$PDK_ROOT"


@dataclass
class PDKValidationResult:
    """Result of PDK sanity check."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_pdk(pdk: PDKConfig) -> PDKValidationResult:
    """Check PDK configuration for obvious errors."""
    errors = []
    warnings = []

    if pdk.pdk_type != PDKType.CUSTOM:
        if not pdk.liberty_file:
            errors.append("liberty_file is empty")
        if not pdk.lef_file:
            errors.append("lef_file is empty")
        if not pdk.tech_lef:
            errors.append("tech_lef is empty")

    if pdk.clock_period_ns <= 0:
        errors.append(f"clock_period_ns must be positive, got {pdk.clock_period_ns}")
    if pdk.voltage_v <= 0:
        errors.append(f"voltage_v must be positive, got {pdk.voltage_v}")
    if pdk.metal_layers < 3:
        warnings.append(f"only {pdk.metal_layers} metal layers — may limit routing")

    return PDKValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_pdk_installation(
    pdk: PDKConfig,
    pdk_root: Optional[str] = None,
    require_signoff: bool = False,
) -> PDKValidationResult:
    """Check whether the resolved open-source PDK files are present locally."""
    base = validate_pdk(pdk)
    errors = list(base.errors)
    warnings = list(base.warnings)

    resolution = OpenSourcePDKResolver.resolve(pdk, pdk_root=pdk_root, require_existing=True)
    for name in resolution.missing_required:
        path = resolution.files.required_paths()[name]
        errors.append(f"{name} not found: {path}")
    for name in resolution.missing_optional:
        path = resolution.files.optional_paths()[name]
        message = f"{name} not found: {path}"
        if require_signoff:
            errors.append(message)
        else:
            warnings.append(message)

    return PDKValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
