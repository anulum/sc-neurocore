import type { AnalysisResultMetadata, SimulationRunMetadata } from "./api/client";
import type { EvidenceSummaryItem } from "./components/EvidenceSummaryStrip";

export function buildSimulationEvidenceItems(metadata: SimulationRunMetadata): EvidenceSummaryItem[] {
  return [
    { label: "class", value: metadata.evidence_classification },
    { label: "source", value: metadata.source },
    { label: "in", value: metadata.input_sha256.slice(0, 10) },
    { label: "out", value: metadata.result_sha256.slice(0, 10) },
  ];
}

export function buildAnalysisEvidenceItems(metadata: AnalysisResultMetadata): EvidenceSummaryItem[] {
  return [
    { label: "type", value: metadata.analysis_type },
    { label: "class", value: metadata.evidence_classification },
    { label: "source", value: metadata.source },
    { label: "in", value: metadata.input_sha256.slice(0, 10) },
    { label: "out", value: metadata.result_sha256.slice(0, 10) },
  ];
}
