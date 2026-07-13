# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC deck bundle orchestration and evidence manifests

"""Orchestrate deterministic ASIC deck bundles and evidence manifests."""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sc_neurocore.asic_flow.decks import (
    FloorplanGenerator,
    GDSIIExporter,
    PlaceRouteGenerator,
    SDCGenerator,
    SynthesisGenerator,
)
from sc_neurocore.asic_flow.design import DesignParams
from sc_neurocore.asic_flow.estimation import DesignEstimate, PreSynthEstimator
from sc_neurocore.asic_flow.pdk import (
    OpenSourcePDKResolver,
    PDKConfig,
    PDKResolution,
    PDKType,
)
from sc_neurocore.asic_flow.signoff import SignoffGenerator


@dataclass
class ASICFlowOutput:
    """Complete output of the ASIC tape-out flow."""

    synth_tcl: str
    sdc: str
    floorplan_tcl: str
    pnr_tcl: str
    sta_tcl: str
    drc_script: str
    lvs_script: str
    gdsii_script: str
    makefile: str
    filelist: List[str]

    def to_dict(self) -> Dict[str, str]:
        """Map canonical bundle filenames to their generated contents."""
        return {
            "synth.tcl": self.synth_tcl,
            "constraints.sdc": self.sdc,
            "floorplan.tcl": self.floorplan_tcl,
            "pnr.tcl": self.pnr_tcl,
            "sta.tcl": self.sta_tcl,
            "drc_check.py": self.drc_script,
            "lvs_check.sh": self.lvs_script,
            "gdsii_export.sh": self.gdsii_script,
            "Makefile": self.makefile,
        }


class ASICFlowGenerator:
    """Top-level generator for the complete ASIC tape-out pipeline."""

    def generate(
        self,
        pdk: PDKConfig,
        design: DesignParams,
    ) -> ASICFlowOutput:
        """Generate all deterministic decks for one PDK/design pair.

        Parameters
        ----------
        pdk:
            Resolved process configuration used by every deck.
        design:
            Logical and physical design parameters.

        Returns
        -------
        ASICFlowOutput
            Nine generated files without executing external EDA tools.
        """
        synth = SynthesisGenerator.generate(pdk, design)
        sdc = SDCGenerator.generate(pdk, design)
        fp = FloorplanGenerator.generate(pdk, design)
        pnr = PlaceRouteGenerator.generate(pdk, design)
        sta = SignoffGenerator.generate_sta_script(pdk, design)
        drc = SignoffGenerator.generate_drc_script(pdk, design)
        lvs = SignoffGenerator.generate_lvs_script(pdk, design)
        gdsii = GDSIIExporter.generate(pdk, design)
        makefile = self._generate_makefile(design)

        filelist = list(
            ASICFlowOutput(synth, sdc, fp, pnr, sta, drc, lvs, gdsii, makefile, []).to_dict().keys()
        )

        return ASICFlowOutput(synth, sdc, fp, pnr, sta, drc, lvs, gdsii, makefile, filelist)

    def _generate_makefile(self, design: DesignParams) -> str:
        return textwrap.dedent(f"""\
# SC-NeuroCore ASIC Flow — Makefile
# Usage: make all

TOP = {design.top_module}

.PHONY: all synth floorplan pnr sta drc lvs gdsii clean

all: synth floorplan pnr sta drc lvs gdsii

synth:
\tyosys -c synth.tcl 2>&1 | tee logs/synth.log

floorplan: synth
\topenroad -exit floorplan.tcl 2>&1 | tee logs/floorplan.log

pnr: floorplan
\topenroad -exit pnr.tcl 2>&1 | tee logs/pnr.log

sta: pnr
\tsta sta.tcl 2>&1 | tee logs/sta.log

drc: gdsii
\tpython3 drc_check.py 2>&1 | tee logs/drc.log

lvs: pnr
\tbash lvs_check.sh 2>&1 | tee logs/lvs.log

gdsii: pnr
\tbash gdsii_export.sh 2>&1 | tee logs/gdsii.log

clean:
\trm -rf synth_$(TOP).* $(TOP)_final.* $(TOP).gds logs/
""")


@dataclass(frozen=True)
class ASICFlowBundle:
    """Generated ASIC flow files plus the evidence manifest path."""

    output_dir: str
    manifest_path: str
    file_paths: Dict[str, str]
    pdk_resolution: PDKResolution
    estimate: DesignEstimate

    def to_dict(self) -> Dict[str, Any]:
        """Serialise bundle paths, PDK resolution, and screening estimate."""
        return {
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "file_paths": dict(self.file_paths),
            "pdk_resolution": {
                "pdk": _pdk_to_manifest(self.pdk_resolution.pdk),
                "files": asdict(self.pdk_resolution.files),
                "missing_required": list(self.pdk_resolution.missing_required),
                "missing_optional": list(self.pdk_resolution.missing_optional),
                "usable_for_synthesis": self.pdk_resolution.usable_for_synthesis,
                "usable_for_signoff": self.pdk_resolution.usable_for_signoff,
            },
            "estimate": asdict(self.estimate),
        }


def generate_asic_flow_bundle(
    output_dir: str | Path,
    *,
    pdk_type: PDKType | str = PDKType.SKY130,
    design: Optional[DesignParams] = None,
    pdk_root: Optional[str] = None,
    require_pdk_files: bool = False,
    n_neurons: int = 16,
    n_synapses: int = 256,
    bitstream_width: int = 256,
    n_aer_ports: int = 4,
    formal_evidence_artifacts: Optional[List[str]] = None,
) -> ASICFlowBundle:
    """Write a complete ASIC flow deck and evidence manifest in one call.

    The helper deliberately does not run Yosys/OpenROAD. It materialises the
    scripts, resolves the requested PDK paths, records missing artefacts, and
    adds a pre-synthesis estimate so Python API users can inspect the bundle
    before launching external EDA tools.
    """
    pdk_enum = _normalise_pdk_type(pdk_type)
    design = design or DesignParams()
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    pdk = PDKConfig.from_pdk_type(pdk_enum)
    resolution = OpenSourcePDKResolver.resolve(
        pdk,
        pdk_root=pdk_root,
        require_existing=require_pdk_files,
    )
    flow = ASICFlowGenerator().generate(resolution.pdk, design)

    file_paths: Dict[str, str] = {}
    for name, content in flow.to_dict().items():
        path = out / name
        path.write_text(content, encoding="utf-8")
        file_paths[name] = str(path)

    estimate = PreSynthEstimator.estimate(
        n_neurons=n_neurons,
        n_synapses=n_synapses,
        bitstream_width=bitstream_width,
        n_aer_ports=n_aer_ports,
        pdk=resolution.pdk,
    )
    manifest = _build_asic_flow_manifest(
        design=design,
        pdk_resolution=resolution,
        estimate=estimate,
        file_paths=file_paths,
        require_pdk_files=require_pdk_files,
        formal_evidence_artifacts=formal_evidence_artifacts or [],
    )
    manifest_path = out / "asic_flow_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return ASICFlowBundle(
        output_dir=str(out),
        manifest_path=str(manifest_path),
        file_paths=file_paths,
        pdk_resolution=resolution,
        estimate=estimate,
    )


def _normalise_pdk_type(pdk_type: PDKType | str) -> PDKType:
    """Convert a process enum or case-insensitive value into ``PDKType``."""
    if isinstance(pdk_type, PDKType):
        return pdk_type
    try:
        return PDKType(str(pdk_type).lower())
    except ValueError as exc:
        valid = ", ".join(p.value for p in PDKType)
        raise ValueError(f"unknown PDK type {pdk_type!r}; expected one of: {valid}") from exc


def _pdk_to_manifest(pdk: PDKConfig) -> Dict[str, Any]:
    """Convert a PDK configuration into JSON-compatible manifest fields."""
    data = asdict(pdk)
    data["pdk_type"] = pdk.pdk_type.value
    data["is_open_source"] = pdk.is_open_source
    return data


def _design_to_manifest(design: DesignParams) -> Dict[str, Any]:
    """Convert design inputs and derived geometry into manifest fields."""
    data = asdict(design)
    data["clock_period_ns"] = design.clock_period_ns
    data["die_width_um"] = design.die_width_um
    data["die_height_um"] = design.die_height_um
    data["core_area_mm2"] = design.core_area_mm2
    return data


def _build_asic_flow_manifest(
    *,
    design: DesignParams,
    pdk_resolution: PDKResolution,
    estimate: DesignEstimate,
    file_paths: Dict[str, str],
    require_pdk_files: bool,
    formal_evidence_artifacts: List[str],
) -> Dict[str, Any]:
    """Build the evidence manifest without making physical-design claims."""
    formal_status = _formal_evidence_status(formal_evidence_artifacts)
    return {
        "schema": "sc-neurocore.asic_flow_manifest.v1",
        "claim_status": {
            "scripts_generated": True,
            "external_eda_executed": False,
            "physical_ppa_claim_allowed": False,
            "formal_evidence_attached": formal_status["attached"],
            "formal_evidence_complete_for_claim": formal_status["complete_for_claim"],
            "reason": (
                "Generated decks and pre-synthesis estimates only; quote physical "
                "area, power, timing, or GDSII claims only after attaching exact "
                "OpenROAD/container and PDK revision evidence."
            ),
        },
        "require_pdk_files": require_pdk_files,
        "pdk": _pdk_to_manifest(pdk_resolution.pdk),
        "pdk_files": asdict(pdk_resolution.files),
        "missing_required": list(pdk_resolution.missing_required),
        "missing_optional": list(pdk_resolution.missing_optional),
        "usable_for_synthesis": pdk_resolution.usable_for_synthesis,
        "usable_for_signoff": pdk_resolution.usable_for_signoff,
        "formal_evidence": formal_status,
        "design": _design_to_manifest(design),
        "estimate": asdict(estimate),
        "generated_files": dict(sorted(file_paths.items())),
    }


def _formal_evidence_status(formal_evidence_artifacts: List[str]) -> Dict[str, Any]:
    """Classify attached proof sources and machine-readable proof reports."""
    artifacts = sorted(str(Path(item)) for item in formal_evidence_artifacts)
    has_proof_script = any(item.endswith((".sby", ".sv", ".sva")) for item in artifacts)
    has_report = any(item.endswith((".json", ".txt", ".log")) for item in artifacts)
    return {
        "artifacts": artifacts,
        "attached": bool(artifacts),
        "complete_for_claim": bool(artifacts) and has_proof_script and has_report,
        "required_types_for_claim": {
            "proof_script": [".sby", ".sv", ".sva"],
            "report_or_log": [".json", ".txt", ".log"],
        },
    }
