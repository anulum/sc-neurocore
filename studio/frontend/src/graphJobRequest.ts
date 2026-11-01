// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Immutable graph request and local edit fencing

import type { NetworkGraph } from "./api/types";
import { canonicalSealText } from "./evidenceSeal";
import { stringifyStudioJson } from "./studioJson";

/** Exact wire bytes and a local edit fence; the key is not the server's graph digest. */
export interface GraphJobRequestSnapshot {
  json: string;
  inputKey: string;
}

/**
 * Omit absent optional inputs without inventing a server default.
 *
 * @param fields - Known graph input fields.
 * @returns Only the explicitly supplied values.
 */
function supplied(fields: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(fields).filter(([, value]) => value !== undefined));
}

/**
 * Fence scientific edits while preserving canvas layout independence.
 *
 * This compares supplied inputs only. Server revalidation resolves defaults and
 * supplies the actual mode-bound scientific digest before any result adoption.
 *
 * @param graph - Current graph and experiment settings.
 * @returns A local canonical input key, excluding positions and drawing metadata.
 */
export function graphJobInputKey(graph: NetworkGraph): string {
  return canonicalSealText(supplied({
    populations: graph.populations.map((population) => supplied({
      id: population.id, label: population.label, model: population.model,
      count: population.count, neuron_type: population.neuron_type, params: population.params,
      drive: population.drive === undefined ? undefined : supplied({ ...population.drive }),
    })),
    projections: graph.projections.map((projection) => supplied({
      id: projection.id, source: projection.source, target: projection.target,
      weight: projection.weight, delay: projection.delay, rule: projection.rule,
      probability: projection.probability, seed: projection.seed, autapses: projection.autapses,
    })),
    duration: graph.duration, dt: graph.dt, seed: graph.seed,
  }));
}

/**
 * Capture a submission before an asynchronous operation can observe later edits.
 *
 * @param graph - Current typed graph; all scientific numbers must be finite.
 * @returns Immutable JSON text and its local edit fence, never references to mutable nodes.
 */
export function snapshotGraphJobRequest(graph: NetworkGraph): GraphJobRequestSnapshot {
  const inputKey = graphJobInputKey(graph);
  return { json: stringifyStudioJson(graph), inputKey };
}
