// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio training export helpers

/**
 * Writing training checkpoints and restore verifications out of the browser.
 *
 * An export here is a plan rather than an act: the caller is handed a blob, a
 * filename, and a `writeExport` it may never call. That lets a view render the
 * name of the file it would write, and lets a test read the bytes without a
 * download ever starting.
 *
 * The plan is also how "there is nothing to export" is expressed. A missing
 * restore verification produces an unavailable plan carrying the sentence to
 * show, not a thrown error and not a plan that writes an empty file.
 */

import type {
  TrainingCheckpointPayload,
  TrainingWeightRestorePlan,
} from "./api/client";
import { downloadBrowserArtefact } from "./browserArtefactDownload";
import {
  buildTrainingWeightRestoreVerificationManifest,
  type TrainingWeightRestoreVerification,
} from "./trainingRestore";

/**
 * How an export reaches the reader's disk.
 *
 * Injected so the tests can watch a download without one happening.
 */
export type StudioTrainingExportDownloader = (payload: Blob, filename: string) => void;

/** The bytes of an export and the name to save them under. */
export interface StudioTrainingExport {
  blob: Blob;
  filename: string;
}

/** An export that exists, with the call that writes it. */
export interface StudioTrainingExportReadyPlan {
  available: true;
  export: StudioTrainingExport;
  writeExport: (downloader?: StudioTrainingExportDownloader) => void;
}

/** No export, and the sentence explaining why, ready to show. */
export interface StudioTrainingExportUnavailablePlan {
  available: false;
  message: string;
}

/** An export that can be written, or the reason it cannot. */
export type StudioTrainingExportPlan =
  | StudioTrainingExportReadyPlan
  | StudioTrainingExportUnavailablePlan;

/**
 * Reduce an identifier to characters a filename can carry.
 *
 * @param value - The identifier, as the server gave it.
 * @returns The reduced identifier, or `training` when nothing survives.
 */
function safeTrainingExportId(value: string): string {
  const safeValue = value.trim().replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "");
  return safeValue.length > 0 ? safeValue : "training";
}

/**
 * Serialise a value into an indented JSON blob.
 *
 * @param value - The value.
 * @returns The blob.
 */
function jsonBlob(value: unknown): Blob {
  return new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
}

/**
 * Name the file a checkpoint is saved as.
 *
 * @param checkpoint - The checkpoint.
 * @returns The filename.
 */
export function trainingCheckpointFilename(checkpoint: TrainingCheckpointPayload): string {
  return `training_checkpoint_${safeTrainingExportId(checkpoint.job_id)}.json`;
}

/**
 * Build the export for a training checkpoint.
 *
 * @param checkpoint - The checkpoint.
 * @returns Its bytes and filename.
 */
export function trainingCheckpointExport(
  checkpoint: TrainingCheckpointPayload,
): StudioTrainingExport {
  return {
    blob: jsonBlob(checkpoint),
    filename: trainingCheckpointFilename(checkpoint),
  };
}

/**
 * Build a ready plan that writes a training checkpoint.
 *
 * A checkpoint in hand is always exportable, so this plan is never the
 * unavailable one.
 *
 * @param checkpoint - The checkpoint.
 * @returns The plan.
 */
export function trainingCheckpointExportPlan(
  checkpoint: TrainingCheckpointPayload,
): StudioTrainingExportReadyPlan {
  const exported = trainingCheckpointExport(checkpoint);
  return readyTrainingExportPlan(exported);
}

/**
 * Build the export recording that restored weights were verified.
 *
 * @param restorePlan - The restore the verification was made against.
 * @param verification - What the verification found.
 * @returns Its bytes and filename.
 */
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

/**
 * Build the plan for a restore verification, if there is one to export.
 *
 * @param restorePlan - The restore, or `null` when none has been read.
 * @param verification - The verification, or `null` when none has run.
 * @returns A ready plan, or an unavailable one naming what is missing.
 */
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

/**
 * Wrap an export in a plan whose `writeExport` downloads it.
 *
 * @param exported - The export.
 * @returns The plan. Its `writeExport` uses the browser download unless
 *   the caller passes another downloader.
 */
function readyTrainingExportPlan(exported: StudioTrainingExport): StudioTrainingExportReadyPlan {
  return {
    available: true,
    export: exported,
    writeExport: (downloader = downloadBrowserArtefact) => {
      downloader(exported.blob, exported.filename);
    },
  };
}
