// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * The evidence strip that sits under a plot.
 *
 * Each item is a short label and a value, sized to be read at a glance beside
 * the figure it describes. The digests are shown as ten characters: enough to
 * compare two runs by eye, and short enough not to crowd the strip. The full
 * digests stay in the exported bundle, which is where a real check is made.
 */

import type { AnalysisResultMetadata, SimulationRunMetadata } from "./api/client";
import type { EvidenceSummaryItem } from "./components/EvidenceSummaryStrip";

/**
 * Describe a simulation run for the strip under its plot.
 *
 * @param metadata - The run's metadata.
 * @returns The items to show, in reading order.
 */
export function buildSimulationEvidenceItems(metadata: SimulationRunMetadata): EvidenceSummaryItem[] {
  return [
    { label: "class", value: metadata.evidence_classification },
    { label: "source", value: metadata.source },
    { label: "status", value: metadata.status },
    { label: "in", value: metadata.input_sha256.slice(0, 10) },
    { label: "out", value: metadata.result_sha256.slice(0, 10) },
  ];
}

/**
 * Describe an analysis result for the strip under its plot.
 *
 * @param metadata - The result's metadata.
 * @returns The items to show, in reading order.
 */
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
