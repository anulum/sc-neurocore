// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import type { AnalysisResultMetadata, SimulationRunMetadata } from "./api/client";
import type { EvidenceSummaryItem } from "./components/EvidenceSummaryStrip";

export function buildSimulationEvidenceItems(metadata: SimulationRunMetadata): EvidenceSummaryItem[] {
  return [
    { label: "class", value: metadata.evidence_classification },
    { label: "source", value: metadata.source },
    { label: "status", value: metadata.status },
    { label: "in", value: metadata.input_sha256.slice(0, 10) },
    { label: "out", value: metadata.result_sha256.slice(0, 10) },
  ];
}

export function buildAnalysisEvidenceItems(metadata: AnalysisResultMetadata): EvidenceSummaryItem[] {
  return [
    { label: "type", value: metadata.analysis_type },
    { label: "class", value: metadata.evidence_classification },
    { label: "source", value: metadata.source },
    { label: "status", value: metadata.status },
    { label: "in", value: metadata.input_sha256.slice(0, 10) },
    { label: "out", value: metadata.result_sha256.slice(0, 10) },
  ];
}
