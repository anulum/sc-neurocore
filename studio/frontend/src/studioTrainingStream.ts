// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio training stream event parsing

/**
 * Reading a training run's event stream.
 *
 * Every frame arrives as text from a server this build does not version
 * against, so each is parsed defensively: a frame that does not parse, or
 * carries a shape this build does not know, is dropped rather than thrown.
 * A dropped frame costs one update; a thrown one would take down the stream
 * and with it every update after it.
 *
 * The event source is created through a factory so a test can drive the
 * parsing without a network.
 */

import type { TrainingEpochMetrics } from "./api/client";

/** The two ways a run ends of its own accord. */
export type StudioTrainingTerminalStatus = "completed" | "stopped";

/** One frame, as the browser delivers it. */
export type StudioTrainingStreamMessageEvent = MessageEvent<string>;

/** The part of an `EventSource` this module uses, so a test can supply its own. */
export interface StudioTrainingStreamEventSource {
  close: () => void;
  onerror: ((event: Event) => void) | null;
  onmessage: ((event: StudioTrainingStreamMessageEvent) => void) | null;
}

/** How an event source is created, so a test can supply one that is not a network. */
export type StudioTrainingStreamFactory = (url: string) => StudioTrainingStreamEventSource;

/** What the caller wants done with each kind of update. */
export interface StudioTrainingStreamHandlers {
  onDisconnected: () => void;
  onEpoch: (metrics: TrainingEpochMetrics) => void;
  onError: (message: string) => void;
  onTerminal: (status: StudioTrainingTerminalStatus) => void;
}

/** One understood update: an epoch, a terminal status, or an error. */
export type StudioTrainingStreamUpdate =
  | { kind: "epoch"; metrics: TrainingEpochMetrics }
  | { kind: "terminal"; status: StudioTrainingTerminalStatus }
  | { kind: "error"; message: string };

/**
 * The stream URL for one job.
 *
 * @param jobId - The job to follow.
 * @returns The URL.
 */
export function studioTrainingStreamUrl(jobId: string): string {
  return `/api/training/stream/${encodeURIComponent(jobId)}`;
}

/**
 * Follow one training run's stream.
 *
 * @param jobId - The job to follow.
 * @param handlers - What to do with each update.
 * @param createEventSource - How to create the source; a test supplies its own.
 * @returns The source, for the caller to close.
 */
export function connectStudioTrainingEventSource(
  jobId: string,
  handlers: StudioTrainingStreamHandlers,
  createEventSource: StudioTrainingStreamFactory = defaultStudioTrainingStreamFactory,
): StudioTrainingStreamEventSource {
  const eventSource = createEventSource(studioTrainingStreamUrl(jobId));
  eventSource.onmessage = (event) => {
    const update = parseStudioTrainingStreamMessage(event.data);
    if (update === null) {
      return;
    }
    if (update.kind === "epoch") {
      handlers.onEpoch(update.metrics);
      return;
    }
    if (update.kind === "terminal") {
      handlers.onTerminal(update.status);
      eventSource.close();
      return;
    }
    handlers.onError(update.message);
    eventSource.close();
  };
  eventSource.onerror = () => {
    handlers.onDisconnected();
    eventSource.close();
  };
  return eventSource;
}

/**
 * Read one frame, or say it could not be read.
 *
 * @param data - The frame's text.
 * @returns The update, or `null` for a frame this build cannot use.
 */
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

/**
 * Read an epoch's metrics, or say the frame did not carry any.
 *
 * @param value - The frame's payload.
 * @returns The metrics, or `null`.
 */
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

/**
 * Read a plain object, or an empty one.
 *
 * @param value - The value.
 * @returns The object.
 */
function recordValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

/**
 * Read a string, falling back when it is not one.
 *
 * @param value - The value.
 * @param fallback - What to use instead.
 * @returns The string.
 */
function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" && value.length > 0 ? value : fallback;
}

/**
 * Read a finite number, or zero.
 *
 * @param value - The value.
 * @returns The number.
 */
function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * Read a record of finite numbers, dropping entries that are not.
 *
 * @param value - The value.
 * @returns The record.
 */
function numberRecordValue(value: unknown): Record<string, number> {
  return Object.fromEntries(
    Object.entries(recordValue(value)).filter((entry): entry is [string, number] =>
      finiteNumber(entry[1])),
  );
}

/**
 * Create a real `EventSource`, which is what production uses.
 *
 * @param url - The stream URL.
 * @returns The source.
 */
function defaultStudioTrainingStreamFactory(url: string): StudioTrainingStreamEventSource {
  return new EventSource(url);
}
