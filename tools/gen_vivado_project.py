#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - UltraScale+ Vivado project generator

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UltraScalePlusSku:
    name: str
    part: str
    lut_budget: int
    ff_budget: int
    dsp_budget: int
    bram_36k_budget: int
    uram_budget: int


SUPPORTED_SKUS: dict[str, UltraScalePlusSku] = {
    "zu3eg": UltraScalePlusSku(
        name="ZU3EG",
        part="xczu3eg-sbva484-1-e",
        lut_budget=70_560,
        ff_budget=141_120,
        dsp_budget=360,
        bram_36k_budget=216,
        uram_budget=0,
    ),
    "zu9eg": UltraScalePlusSku(
        name="ZU9EG",
        part="xczu9eg-ffvb1156-2-e",
        lut_budget=274_080,
        ff_budget=548_160,
        dsp_budget=2_520,
        bram_36k_budget=912,
        uram_budget=0,
    ),
}


@dataclass(frozen=True)
class VivadoManifest:
    top: str
    sku: UltraScalePlusSku
    clock_mhz: int
    sources: tuple[Path, ...]
    xdc: tuple[Path, ...]
    output_dir: Path

    @property
    def clock_period_ns(self) -> float:
        if self.clock_mhz <= 0:
            raise ValueError("clock_mhz must be positive")
        return 1000.0 / float(self.clock_mhz)


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Vivado manifest must be a JSON object")
    return payload


def _paths(root: Path, values: Any, field_name: str) -> tuple[Path, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list")
    paths: list[Path] = []
    for raw in values:
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{field_name} entries must be non-empty strings")
        path = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"{field_name} entry does not exist: {path}")
        paths.append(path)
    return tuple(paths)


def load_manifest(path: Path) -> VivadoManifest:
    payload = _read_manifest(path)
    root = path.parent.resolve()
    top = payload.get("top")
    if not isinstance(top, str) or not top.isidentifier():
        raise ValueError("top must be a valid Verilog identifier")
    sku_key = payload.get("sku")
    if not isinstance(sku_key, str) or sku_key.lower() not in SUPPORTED_SKUS:
        raise ValueError(f"sku must be one of {sorted(SUPPORTED_SKUS)}")
    clock_mhz = payload.get("clock_mhz", 250)
    if not isinstance(clock_mhz, int) or clock_mhz <= 0:
        raise ValueError("clock_mhz must be a positive integer")
    output_dir_raw = payload.get("output_dir", "out/ultrascale_plus")
    if not isinstance(output_dir_raw, str) or not output_dir_raw:
        raise ValueError("output_dir must be a non-empty string")
    return VivadoManifest(
        top=top,
        sku=SUPPORTED_SKUS[sku_key.lower()],
        clock_mhz=clock_mhz,
        sources=_paths(root, payload.get("sources"), "sources"),
        xdc=_paths(root, payload.get("xdc"), "xdc"),
        output_dir=(root / output_dir_raw).resolve()
        if not Path(output_dir_raw).is_absolute()
        else Path(output_dir_raw),
    )


def generate_tcl(manifest: VivadoManifest) -> str:
    lines = [
        "# SPDX-License-Identifier: AGPL-3.0-or-later",
        "# Commercial license available",
        "# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.",
        "# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.",
        "# ORCID: 0009-0009-3560-0851",
        "# Contact: www.anulum.li | protoscience@anulum.li",
        "# SC-NeuroCore - generated UltraScale+ Vivado batch project",
        "set_msg_config -id {Common 17-55} -new_severity ERROR",
        f"set TOP {manifest.top}",
        f"set PART {manifest.sku.part}",
        f"set CLOCK_MHZ {manifest.clock_mhz}",
        f"set CLOCK_PERIOD_NS {manifest.clock_period_ns:.6f}",
        f"set OUT_DIR {manifest.output_dir.as_posix()}",
        "file mkdir $OUT_DIR",
        "create_project -in_memory -part $PART sc_neurocore_ultrascale_plus",
        "set_property target_language Verilog [current_project]",
    ]
    for source in manifest.sources:
        lines.append(f"read_verilog -sv {source.as_posix()}")
    for xdc in manifest.xdc:
        lines.append(f"read_xdc {xdc.as_posix()}")
    lines.extend(
        [
            "synth_design -top $TOP -part $PART -mode out_of_context",
            "opt_design",
            "place_design",
            "route_design",
            "report_utilization -file $OUT_DIR/${TOP}_utilisation.rpt",
            "report_timing_summary -file $OUT_DIR/${TOP}_timing.rpt",
            "write_checkpoint -force $OUT_DIR/${TOP}.dcp",
            "write_bitstream -force $OUT_DIR/${TOP}.bit",
        ]
    )
    return "\n".join(lines) + "\n"


def sku_baseline() -> dict[str, dict[str, int | str]]:
    return {
        key: {
            "name": sku.name,
            "part": sku.part,
            "lut_budget": sku.lut_budget,
            "ff_budget": sku.ff_budget,
            "dsp_budget": sku.dsp_budget,
            "bram_36k_budget": sku.bram_36k_budget,
            "uram_budget": sku.uram_budget,
            "dsp_primitive": "DSP48E2",
        }
        for key, sku in SUPPORTED_SKUS.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate UltraScale+ Vivado batch Tcl.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = load_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate_tcl(manifest), encoding="utf-8")
    if args.baseline_json is not None:
        args.baseline_json.parent.mkdir(parents=True, exist_ok=True)
        args.baseline_json.write_text(
            json.dumps(sku_baseline(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "top": manifest.top,
                "sku": manifest.sku.name,
                "part": manifest.sku.part,
                "clock_mhz": manifest.clock_mhz,
                "sources": len(manifest.sources),
                "xdc": len(manifest.xdc),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
