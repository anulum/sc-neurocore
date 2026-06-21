// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio training export helpers

import type {
  TrainingCheckpointPayload,
  TrainingWeightRestorePlan,
} from "./api/client";
import {
  buildTrainingWeightRestoreVerificationManifest,
  type TrainingWeightRestoreVerification,
} from "./trainingRestore";

export interface StudioTrainingExport {
  blob: Blob;
  filename: string;
}

function safeTrainingExportId(value: string): string {
  const safeValue = value.trim().replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "");
  return safeValue.length > 0 ? safeValue : "training";
}

function jsonBlob(value: unknown): Blob {
  return new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
}

export function trainingCheckpointFilename(checkpoint: TrainingCheckpointPayload): string {
  return `training_checkpoint_${safeTrainingExportId(checkpoint.job_id)}.json`;
}

export function trainingCheckpointExport(
  checkpoint: TrainingCheckpointPayload,
): StudioTrainingExport {
  return {
    blob: jsonBlob(checkpoint),
    filename: trainingCheckpointFilename(checkpoint),
  };
}

export function trainingWeightRestoreVerificationExport(
  restorePlan: TrainingWeightRestorePlan,
  verification: TrainingWeightRestoreVerification,
): StudioTrainingExport {
  const manifest = buildTrainingWeightRestoreVerificationManifest(restorePlan, verification);
  return {
    blob: jsonBlob(manifest),
    filename: `training_weight_restore_${safeTrainingExportId(manifest.source_job_id)}.json`,
  };
}
