#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — isolated ASIC-flow benchmark probe

"""Emit one cold-process ASIC-flow timing and fidelity sample as JSON."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys
import tempfile
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sc_neurocore.asic_flow.asic_flow import DesignParams


def _design() -> DesignParams:
    """Return the deterministic benchmark design."""
    from sc_neurocore.asic_flow.asic_flow import DesignParams, SCASICOptimisationConfig

    return DesignParams(
        top_module="edge_snn",
        rtl_files=["rtl/edge_snn.sv", "rtl/router.sv"],
        target_frequency_mhz=125.0,
        utilisation=0.57,
        sc_optimisation=SCASICOptimisationConfig(max_fanout=12),
    )


def _generated_payload() -> dict[str, object]:
    """Generate the deterministic in-memory ASIC deck workload."""
    from sc_neurocore.asic_flow.asic_flow import (
        ASICFlowGenerator,
        CDCCheckGenerator,
        HierarchicalFlow,
        IOConstraintGenerator,
        IRDropGenerator,
        LECGenerator,
        MultiCornerAnalysis,
        OCVConfig,
        PDKConfig,
        PDKType,
        PreSynthEstimator,
    )

    pdk = PDKConfig.from_pdk_type(PDKType.SKY130).with_pdk_root("/opt/pdks")
    design = _design()
    flow = ASICFlowGenerator().generate(pdk, design)
    hierarchy = HierarchicalFlow(top_design=design)
    return {
        "flow": flow.to_dict(),
        "filelist": flow.filelist,
        "estimate": asdict(PreSynthEstimator.estimate(32, 512, 128, 8, pdk)),
        "corners": MultiCornerAnalysis.generate(pdk, design),
        "cdc": CDCCheckGenerator.generate(design, ["clk", "aer_clk"]),
        "ir": IRDropGenerator.generate(pdk, design, 0.17),
        "io": IOConstraintGenerator.generate(
            IOConstraintGenerator.auto_assign(["clk", "rst_n", "aer_in", "aer_out"]),
            design,
        ),
        "lec": LECGenerator.generate(design),
        "ocv": OCVConfig.conservative().generate_sdc_fragment(),
        "hierarchy": hierarchy.generate_top_integration(pdk),
    }


def _measure_decks() -> tuple[int, bytes]:
    """Return median deck latency and canonical serialised output bytes."""
    samples: list[int] = []
    payload: dict[str, object] | None = None
    for _ in range(25):
        started = time.perf_counter_ns()
        payload = _generated_payload()
        samples.append(time.perf_counter_ns() - started)
    if payload is None:
        raise RuntimeError("deck workload did not execute")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return int(statistics.median(samples)), encoded


def _measure_bundles() -> tuple[int, int, str]:
    """Return median bundle latency, file count, and manifest schema."""
    from sc_neurocore.asic_flow.asic_flow import generate_asic_flow_bundle

    samples: list[int] = []
    file_count = 0
    schema = ""
    for _ in range(3):
        with tempfile.TemporaryDirectory(prefix="sc-neurocore-asic-") as directory:
            started = time.perf_counter_ns()
            bundle = generate_asic_flow_bundle(
                Path(directory),
                pdk_type="sky130",
                design=_design(),
                pdk_root="/opt/pdks",
                n_neurons=32,
                n_synapses=512,
                bitstream_width=128,
                n_aer_ports=8,
                formal_evidence_artifacts=["formal/edge_snn.sby", "formal/report.json"],
            )
            samples.append(time.perf_counter_ns() - started)
            manifest = json.loads(Path(bundle.manifest_path).read_text(encoding="utf-8"))
            file_count = len(bundle.file_paths)
            raw_schema = manifest.get("schema") if isinstance(manifest, dict) else None
            if not isinstance(raw_schema, str):
                raise RuntimeError("ASIC manifest has no schema")
            schema = raw_schema
    return int(statistics.median(samples)), file_count, schema


def main() -> int:
    """Run one probe and write its validated JSON payload to standard output."""
    import_started = time.perf_counter_ns()
    from sc_neurocore.asic_flow import asic_flow as _asic_flow

    import_ns = time.perf_counter_ns() - import_started
    if not hasattr(_asic_flow, "PDKConfig"):
        raise RuntimeError("ASIC-flow module has no PDK configuration surface")
    deck_ns, encoded = _measure_decks()
    bundle_ns, file_count, schema = _measure_bundles()
    if file_count != 9 or schema != "sc-neurocore.asic_flow_manifest.v1":
        raise RuntimeError("ASIC bundle probe produced an invalid contract")
    maximum_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        maximum_rss_kib //= 1024
    print(
        json.dumps(
            {
                "import_ns": import_ns,
                "deck_generation_ns": deck_ns,
                "bundle_write_ns": bundle_ns,
                "max_rss_kib": maximum_rss_kib,
                "generated_sha256": hashlib.sha256(encoded).hexdigest(),
                "generated_bytes": len(encoded),
                "bundle_file_count": file_count,
                "manifest_schema": schema,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
