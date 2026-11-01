// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Complete graph artifact validation before presentation

import type { GraphExecutionBlock, GraphPopulationResult, GraphProjectionTopology, GraphSimResult, MetricContract } from "./api/types";
import { canonicalSealText } from "./evidenceSeal";
import { graphBytesSha256, type GraphJob } from "./graphJobs";
import { parseStudioJson } from "./studioJson";

/** A complete successful result, distinct from an admission or a compact receipt. */
export type GraphJobSimulationResult = Required<Omit<GraphSimResult, "errors">> & { success: true };

/** The complete-result byte ceiling imposed by the graph worker. */
export const GRAPH_RESULT_MAX_BYTES = 16 * 1024 * 1024;

/**
 * Identify a JSON object whose fields can be inspected.
 *
 * @param value - Decoded untrusted field.
 * @returns Whether it is a named-field container.
 */
function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Identify a nonempty textual contract field.
 *
 * @param value - Decoded untrusted field.
 * @returns Whether it carries text without coercion.
 */
function text(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

/**
 * Identify a finite scalar without coercing JSON strings or booleans.
 *
 * @param value - Decoded untrusted field.
 * @returns Whether it is a finite JavaScript number.
 */
function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * Identify an exactly representable nonnegative count.
 *
 * @param value - Decoded untrusted field.
 * @returns Whether count arithmetic can retain its integer value.
 */
function count(value: unknown): value is number {
  return finite(value) && Number.isSafeInteger(value) && value >= 0;
}

/**
 * Identify a graph seed in the server's unsigned 32-bit domain.
 *
 * @param value - Untrusted seed decoded with the maintained JSON reader.
 * @returns Whether it is an integer in the same domain as graph preparation.
 */
function seed(value: unknown): value is number {
  return count(value) && value < 2 ** 32;
}

/**
 * Identify an entire numeric series, retaining its order and length.
 *
 * @param value - Decoded untrusted series.
 * @param accepts - Finite-scalar or integer-count element contract.
 * @returns Whether every element satisfies that contract.
 */
function numbers(value: unknown, accepts: (entry: unknown) => boolean = finite): value is number[] {
  return Array.isArray(value) && value.every(accepts);
}

/**
 * Check the metric interpretation carried alongside the scientific arrays.
 *
 * @param value - Untrusted metric contract.
 * @returns Whether it describes complete network population activity.
 */
function contract(value: unknown): value is MetricContract {
  return record(value) && value.schema_version === "studio.metric-contract.v1"
    && value.kind === "network-population-activity" && value.domain === "complete"
    && text(value.definition) && record(value.units) && Object.values(value.units).every(text)
    && Array.isArray(value.applicability) && value.applicability.every(text)
    && Array.isArray(value.limitations) && value.limitations.every(text) && record(value.domain_detail);
}

/**
 * Check the selected backend and its explicit execution and reproduction semantics.
 *
 * @param value - Untrusted execution block.
 * @returns Whether every displayed execution field has its declared shape.
 */
function execution(value: unknown): value is GraphExecutionBlock {
  return record(value) && record(value.backend) && text(value.backend.selected)
    && Array.isArray(value.backend.rejected) && value.backend.rejected.every((item: unknown) =>
      record(item) && text(item.name) && text(item.reason))
    && [value.loop, value.step_order, value.delay_semantics, value.synapse_semantics,
      value.population_construction, value.autapses, value.state_check].every(text)
    && count(value.projection_latency_steps) && finite(value.network_dt_s) && value.network_dt_s > 0
    && record(value.runtime) && [value.runtime.package_version, value.runtime.python, value.runtime.numpy].every(text);
}

/**
 * Check one population's event domain and paired binned-rate series.
 *
 * @param value - Untrusted population output.
 * @param nSteps - Complete run's checked number of timesteps.
 * @returns Whether events and bins stay within their declared dimensions.
 */
function population(value: unknown, nSteps: number): value is GraphPopulationResult {
  if (!record(value) || ![value.id, value.label, value.model].every(text)
    || !count(value.count) || value.count === 0 || !count(value.offset) || !count(value.n_spikes)
    || value.neuron_type !== "excitatory" && value.neuron_type !== "inhibitory"
    || !finite(value.mean_rate_hz) || value.mean_rate_hz < 0 || !record(value.events) || !record(value.rate)) return false;
  const events = value.events;
  const rate = value.rate;
  const nNeurons = value.count;
  return numbers(events.step, count) && numbers(events.neuron, count)
    && events.step.length === value.n_spikes && events.neuron.length === value.n_spikes
    && events.step.every((step) => step < nSteps) && events.neuron.every((neuron) => neuron < nNeurons)
    && count(rate.bin_steps) && rate.bin_steps > 0 && rate.bin_steps <= nSteps
    && finite(rate.bin_ms) && rate.bin_ms > 0 && count(rate.covered_steps) && rate.covered_steps <= nSteps
    && numbers(rate.time_ms) && numbers(rate.rate_hz)
    && rate.time_ms.length === rate.rate_hz.length && rate.time_ms.length * rate.bin_steps === rate.covered_steps
    && rate.time_ms.every((time) => time >= 0) && rate.rate_hz.every((hz) => hz >= 0);
}

/**
 * Check projection metadata and included CSR bounds without reconstructing connectivity.
 *
 * @param value - Untrusted projection output.
 * @param populations - Checked source and target population dimensions.
 * @param included - Whether this result declares actual connectivity arrays.
 * @returns Whether metadata and any arrays satisfy the declared representation.
 */
function projection(value: unknown, populations: GraphPopulationResult[], included: boolean): value is GraphProjectionTopology {
  if (!record(value) || !text(value.id) || !seed(value.seed) || !finite(value.weight)
    || !count(value.delay_steps) || !text(value.delay_mode) || !count(value.n_synapses) || !count(value.autapses_removed)
    || typeof value.csr_sha256 !== "string" || !/^[0-9a-f]{64}$/.test(value.csr_sha256)
    || value.rule !== "random" && value.rule !== "all_to_all") return false;
  const source = populations.find((item) => item.id === value.source);
  const target = populations.find((item) => item.id === value.target);
  if (source === undefined || target === undefined) return false;
  if (!included) return value.indptr === undefined && value.indices === undefined;
  if (!numbers(value.indptr, count) || !numbers(value.indices, count)) return false;
  const indptr = value.indptr;
  let previous = 0;
  for (const offset of indptr) {
    if (offset < previous) return false;
    previous = offset;
  }
  return indptr.length === source.count + 1 && indptr[0] === 0
    && indptr[indptr.length - 1] === value.n_synapses && value.indices.length === value.n_synapses
    && value.indices.every((index) => index < target.count);
}

/**
 * Check all fields exposed to graph plots, summaries and evidence export.
 *
 * @param value - Entire untrusted result decoded from verified bytes.
 * @returns Whether the successful result's dimensions and declared inputs agree.
 */
function complete(value: unknown): value is GraphJobSimulationResult {
  if (!record(value) || value.success !== true || value.schema_version !== "studio.network-graph-result.v1"
    || !count(value.n_steps) || value.n_steps === 0 || !count(value.n_total) || value.n_total === 0 || !count(value.n_spikes)
    || !finite(value.dt) || value.dt <= 0 || !finite(value.duration) || value.duration !== value.n_steps * value.dt
    || !record(value.spec) || value.spec.schema_version !== "studio.network-graph-spec.v1"
    || value.spec.request_schema_version !== "studio.network-graph.v1" || value.spec.execution_mode !== "async_job"
    || !seed(value.spec.seed)
    || value.spec.dt !== value.dt || value.spec.n_steps !== value.n_steps || value.spec.n_neurons !== value.n_total
    || value.spec.neuron_steps !== value.n_steps * value.n_total || value.spec.duration_effective_ms !== value.duration
    || !execution(value.execution) || !contract(value.contract)
    || !numbers(value.spike_times) || !numbers(value.spike_neurons, count)
    || value.spike_times.length !== value.n_spikes || value.spike_neurons.length !== value.n_spikes
    || !Array.isArray(value.populations) || value.populations.length === 0) return false;
  const nSteps = value.n_steps;
  const nTotal = value.n_total;
  const duration = value.duration;
  const populations: GraphPopulationResult[] = [];
  for (const item of value.populations) {
    if (!population(item, nSteps)) return false;
    populations.push(item);
  }
  if (new Set(populations.map((item) => item.id)).size !== populations.length
    || populations.reduce((sum, item) => sum + item.count, 0) !== nTotal
    || populations.reduce((sum, item) => sum + item.n_spikes, 0) !== value.n_spikes
    || !populations.every((item, index) => item.offset === populations.slice(0, index).reduce((sum, prev) => sum + prev.count, 0))
    || !value.spike_times.every((time) => time >= 0 && time < duration)
    || !value.spike_neurons.every((neuron) => neuron < nTotal)) return false;
  const topology = value.topology;
  const summary = value.graph_summary;
  if (!record(topology) || !count(topology.n_synapses) || typeof topology.connectivity_included !== "boolean"
    || (topology.connectivity_included ? topology.connectivity_omitted_reason !== null : !text(topology.connectivity_omitted_reason))
    || !Array.isArray(topology.projections) || !record(summary)) return false;
  const projections: GraphProjectionTopology[] = [];
  for (const item of topology.projections) {
    if (!projection(item, populations, topology.connectivity_included)) return false;
    projections.push(item);
  }
  return new Set(projections.map((item) => item.id)).size === projections.length
    && projections.reduce((sum, item) => sum + item.n_synapses, 0) === topology.n_synapses
    && summary.n_populations === populations.length && summary.n_projections === projections.length
    && summary.n_neurons === nTotal && summary.n_synapses === topology.n_synapses
    && summary.n_excitatory === populations.filter((item) => item.neuron_type === "excitatory").reduce((sum, item) => sum + item.count, 0)
    && summary.n_inhibitory === populations.filter((item) => item.neuron_type === "inhibitory").reduce((sum, item) => sum + item.count, 0);
}

/**
 * Verify complete bytes against a completed job before decoding scientific data.
 *
 * @param bytes - Complete bounded artifact bytes, not reserialised JSON.
 * @param job - Previously checked actor-owned completed record.
 * @returns The successful graph result bound to its job's receipt and work counts.
 * @throws {Error} When custody, encoding, shape or scientific identity differs.
 */
export async function parseGraphJobResult(bytes: Uint8Array<ArrayBuffer>, job: GraphJob): Promise<GraphJobSimulationResult> {
  if (job.status !== "completed" || job.result === null) throw new Error("Graph job has no complete result.");
  const receipt = { ...job.result };
  const graphSha = job.admission.graph_sha256;
  const captured = bytes.slice();
  if (captured.length === 0 || captured.length > GRAPH_RESULT_MAX_BYTES || captured.length !== receipt.result_size_bytes
    || await graphBytesSha256(captured) !== receipt.result_sha256) throw new Error("Graph result bytes do not match their receipt.");
  const value: unknown = parseStudioJson(new TextDecoder("utf-8", { fatal: true }).decode(captured));
  if (!complete(value) || value.spec.graph_sha256 !== graphSha || receipt.graph_sha256 !== graphSha
    || value.n_steps !== receipt.n_steps || value.n_total !== receipt.n_total || value.n_spikes !== receipt.n_spikes) {
    throw new Error("Graph result does not match its scientific contract.");
  }
  canonicalSealText(value.spec);
  return value;
}
