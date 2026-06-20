import type { ProjectEvidenceModel } from "../projectEvidence";

export default function ProjectEvidenceStrip({ evidence }: { evidence: ProjectEvidenceModel }) {
  return (
    <div style={{
      marginTop: 4,
      padding: 4,
      border: "1px solid var(--border)",
      borderRadius: 4,
      color: "var(--text-muted)",
      fontSize: 9,
      lineHeight: 1.5,
    }}>
      <div>class {evidence.classification}</div>
      <div>name {evidence.name}</div>
      <div>state sha {evidence.stateDigest}</div>
      <div>project sha {evidence.projectDigest}</div>
      <div>{evidence.schemaVersion}</div>
    </div>
  );
}
