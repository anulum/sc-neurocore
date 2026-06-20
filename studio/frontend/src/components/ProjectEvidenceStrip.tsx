import type { ProjectEvidenceModel } from "../projectEvidence";
import EvidenceSummaryStrip from "./EvidenceSummaryStrip";

export default function ProjectEvidenceStrip({ evidence }: { evidence: ProjectEvidenceModel }) {
  return (
    <EvidenceSummaryStrip
      variant="panel"
      items={[
        { label: "class", value: evidence.classification },
        { label: "name", value: evidence.name },
        { label: "state sha", value: evidence.stateDigest },
        { label: "project sha", value: evidence.projectDigest },
        { label: "schema", value: evidence.schemaVersion },
      ]}
    />
  );
}
