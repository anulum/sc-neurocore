// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio training stream event parsing

import type { TrainingEpochMetrics } from "./api/client";

export type StudioTrainingTerminalStatus = "completed" | "stopped";

export type StudioTrainingStreamUpdate =
  | { kind: "epoch"; metrics: TrainingEpochMetrics }
  | { kind: "terminal"; status: StudioTrainingTerminalStatus }
  | { kind: "error"; message: string };

export function parseStudioTrainingStreamMessage(data: string): StudioTrainingStreamUpdate | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(data) as unknown;
  } catch {
    return null;
  }
  const message = recordValue(parsed);
  if (message.event === "epoch") {
    const metrics = trainingEpochMetricsValue(message.data);
    return metrics ? { kind: "epoch", metrics } : null;
  }
  if (message.event === "completed" || message.event === "stopped") {
    return { kind: "terminal", status: message.event };
  }
  if (message.event === "error") {
    const dataRecord = recordValue(message.data);
    return {
      kind: "error",
      message: stringValue(dataRecord.message, "Training failed"),
    };
  }
  return null;
}

function trainingEpochMetricsValue(value: unknown): TrainingEpochMetrics | null {
  const metrics = recordValue(value);
  if (
    !finiteNumber(metrics.epoch)
    || !finiteNumber(metrics.train_loss)
    || !finiteNumber(metrics.train_accuracy)
    || !finiteNumber(metrics.val_loss)
    || !finiteNumber(metrics.val_accuracy)
  ) {
    return null;
  }
  return {
    epoch: metrics.epoch,
    train_loss: metrics.train_loss,
    train_accuracy: metrics.train_accuracy,
    val_loss: metrics.val_loss,
    val_accuracy: metrics.val_accuracy,
    layer_spike_rates: numberRecordValue(metrics.layer_spike_rates),
    param_snapshot: numberRecordValue(metrics.param_snapshot),
  };
}

function recordValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" && value.length > 0 ? value : fallback;
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function numberRecordValue(value: unknown): Record<string, number> {
  return Object.fromEntries(
    Object.entries(recordValue(value)).filter((entry): entry is [string, number] =>
      finiteNumber(entry[1])),
  );
}
