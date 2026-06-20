import type { TrainingConfig, TrainingEpochMetrics } from "./api/client";

export interface TrainingEvidenceModel {
  actionKind: string;
  classification: string;
  configSummary: string;
  evidenceArtifact: string;
  jobId: string;
  latestEpoch: string;
  replayRoute: string;
  status: string;
  statusArtifact: string;
}

export function buildTrainingEvidenceModel(
  jobId: string | null,
  status: string,
  config: Pick<TrainingConfig, "dataset" | "epochs" | "surrogate" | "timesteps">,
  latestEpoch: TrainingEpochMetrics | null,
): TrainingEvidenceModel {
  const submitted = jobId !== null && jobId.length > 0;
  return {
    actionKind: "studio.training.run",
    classification: "training",
    configSummary:
      `${config.dataset}, ${config.epochs} epochs, ${config.surrogate}, ${config.timesteps} steps`,
    evidenceArtifact: submitted ? "training/evidence.json" : "pending",
    jobId: submitted ? jobId : "not submitted",
    latestEpoch: latestEpoch === null ? "none" : String(latestEpoch.epoch),
    replayRoute: "POST /api/training/start",
    status,
    statusArtifact: submitted ? "training/status.json" : "pending",
  };
}
