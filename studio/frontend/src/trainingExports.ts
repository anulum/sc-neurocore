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
import { downloadBrowserArtefact } from "./browserArtefactDownload";
import {
  buildTrainingWeightRestoreVerificationManifest,
  type TrainingWeightRestoreVerification,
} from "./trainingRestore";

export type StudioTrainingExportDownloader = (payload: Blob, filename: string) => void;

export interface StudioTrainingExport {
  blob: Blob;
  filename: string;
}

export interface StudioTrainingExportReadyPlan {
  available: true;
  export: StudioTrainingExport;
  writeExport: (downloader?: StudioTrainingExportDownloader) => void;
}

export interface StudioTrainingExportUnavailablePlan {
  available: false;
  message: string;
}

export type StudioTrainingExportPlan =
  | StudioTrainingExportReadyPlan
  | StudioTrainingExportUnavailablePlan;

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

export function trainingCheckpointExportPlan(
  checkpoint: TrainingCheckpointPayload,
): StudioTrainingExportReadyPlan {
  const exported = trainingCheckpointExport(checkpoint);
  return readyTrainingExportPlan(exported);
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

export function trainingWeightRestoreVerificationExportPlan(
  restorePlan: TrainingWeightRestorePlan | null,
  verification: TrainingWeightRestoreVerification | null,
): StudioTrainingExportPlan {
  if (restorePlan === null || verification === null) {
    return {
      available: false,
      message: "No verified training weight artifact is available for export.",
    };
  }
  const exported = trainingWeightRestoreVerificationExport(restorePlan, verification);
  return readyTrainingExportPlan(exported);
}

function readyTrainingExportPlan(exported: StudioTrainingExport): StudioTrainingExportReadyPlan {
  return {
    available: true,
    export: exported,
    writeExport: (downloader = downloadBrowserArtefact) => {
      downloader(exported.blob, exported.filename);
    },
  };
}
