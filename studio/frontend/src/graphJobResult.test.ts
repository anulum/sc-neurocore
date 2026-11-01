// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Full native graph artifact contract and corruption controls

import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { GRAPH_RESULT_MAX_BYTES, parseGraphJobResult } from "./graphJobResult";
import { graphBytesSha256, parseGraphJob, type GraphJob } from "./graphJobs";

/** Inspect a JSON container while retaining unknown field types. */
function container(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== "object") throw new Error("Missing corpus container");
  return value as Record<string, unknown>;
}

/** Read fresh copies of the actual Python HTTP/worker artifact and its real receipt. */
async function sample(): Promise<{ text: string; bytes: Uint8Array<ArrayBuffer>; job: GraphJob }> {
  const corpus = container(JSON.parse(readFileSync(new URL("./test-data/graphJobContract.json", import.meta.url), "utf8")));
  if (typeof corpus.artifact_text !== "string") throw new Error("Missing native artifact bytes");
  return { text: corpus.artifact_text, bytes: new TextEncoder().encode(corpus.artifact_text), job: await parseGraphJob(corpus.completed, "alice") };
}

/** Rebind only byte custody so schema tests must reach the scientific contract. */
async function rebound(job: GraphJob, bytes: Uint8Array<ArrayBuffer>): Promise<GraphJob> {
  if (job.result === null) throw new Error("Missing native result receipt");
  return { ...job, result: { ...job.result, result_size_bytes: bytes.length, result_sha256: await graphBytesSha256(bytes) } };
}

it("reads the complete real artifact without replacing its selected backend or dropping populations", async () => {
  const native = await sample();
  const result = await parseGraphJobResult(native.bytes, native.job);
  expect(result.n_spikes).toBe(4);
  expect(result.populations.map((pop) => [pop.id, pop.n_spikes])).toEqual([["e", 4], ["i", 0], ["disconnected", 0]]);
  expect(result.execution.backend.selected).toBe("python");
  expect(result.topology.projections).toHaveLength(2);
  expect(result.spec.graph_sha256).toBe(native.job.admission.graph_sha256);
});

it("rejects missing completion and rejects byte tampering even when JSON still parses", async () => {
  const native = await sample();
  await expect(parseGraphJobResult(native.bytes, { ...native.job, status: "failed" })).rejects.toThrow("no complete result");
  await expect(parseGraphJobResult(native.bytes, { ...native.job, result: null })).rejects.toThrow("no complete result");
  await expect(parseGraphJobResult(new Uint8Array(), native.job)).rejects.toThrow("bytes do not match");
  await expect(parseGraphJobResult(new Uint8Array(GRAPH_RESULT_MAX_BYTES + 1), native.job)).rejects.toThrow("bytes do not match");
  await expect(parseGraphJobResult(native.bytes.slice(0, -1), native.job)).rejects.toThrow("bytes do not match");
  const changed = new TextEncoder().encode(native.text.replace('"mean_rate_hz":100.0', '"mean_rate_hz":101.0'));
  await expect(parseGraphJobResult(changed, native.job)).rejects.toThrow("bytes do not match");
});

it.each([
  ["success", false], ["schema_version", "unknown"], ["n_steps", 0], ["n_total", 0], ["n_spikes", -1],
  ["dt", 0], ["duration", 21], ["spec", null], ["spec.schema_version", "unknown"],
  ["spec.request_schema_version", "unknown"], ["spec.execution_mode", "synchronous"],
  ["spec.seed", 2 ** 32],
  ["spec.dt", 1], ["spec.n_steps", 201], ["spec.n_neurons", 5], ["spec.neuron_steps", 801],
  ["spec.duration_effective_ms", 21], ["spec.graph_sha256", "0".repeat(64)],
  ["execution", []], ["execution.backend", null], ["execution.backend.selected", ""],
  ["execution.backend.rejected", null], ["execution.backend.rejected.0", null],
  ["execution.backend.rejected.0.name", ""], ["execution.backend.rejected.0.reason", ""],
  ["execution.loop", 3], ["execution.projection_latency_steps", -1], ["execution.network_dt_s", 0],
  ["execution.runtime", null], ["execution.runtime.python", ""],
  ["contract", []], ["contract.schema_version", "unknown"], ["contract.kind", "other"],
  ["contract.domain", "partial"], ["contract.definition", ""], ["contract.units", null],
  ["contract.units.weight", 0], ["contract.applicability", null], ["contract.applicability.0", 3],
  ["contract.limitations", null], ["contract.limitations.0", false], ["contract.domain_detail", []],
  ["spike_times", null], ["spike_times.0", "6.9"], ["spike_times.0", -1], ["spike_times.0", 20],
  ["spike_times", []], ["spike_neurons", []], ["spike_neurons.0", 0.5], ["spike_neurons.0", 4],
  ["populations", null], ["populations", []], ["populations.0", null], ["populations.0.id", ""],
  ["populations.0.count", 0], ["populations.0.offset", -1], ["populations.0.n_spikes", -1],
  ["populations.0.neuron_type", "unknown"], ["populations.0.mean_rate_hz", -1],
  ["populations.0.events", null], ["populations.0.rate", null], ["populations.0.events.step", null],
  ["populations.0.events.neuron", null], ["populations.0.events.step", []], ["populations.0.events.neuron", []],
  ["populations.0.events.step.0", 200], ["populations.0.events.neuron.0", 2],
  ["populations.0.rate.bin_steps", 0], ["populations.0.rate.bin_steps", 201], ["populations.0.rate.bin_ms", 0],
  ["populations.0.rate.covered_steps", 201], ["populations.0.rate.time_ms", null], ["populations.0.rate.rate_hz", null],
  ["populations.0.rate.time_ms", []], ["populations.0.rate.covered_steps", 199],
  ["populations.0.rate.time_ms.0", -1], ["populations.0.rate.rate_hz.0", -1],
  ["populations.1.id", "e"], ["populations.1.count", 2], ["populations.1.n_spikes", 1], ["populations.1.offset", 0],
  ["topology", null], ["topology.n_synapses", -1], ["topology.connectivity_included", "yes"],
  ["topology.connectivity_omitted_reason", "unexpected"], ["topology.projections", null], ["graph_summary", null],
  ["topology.projections.0", null], ["topology.projections.0.id", ""], ["topology.projections.0.seed", -1],
  ["topology.projections.0.seed", 2 ** 32],
  ["topology.projections.0.weight", null], ["topology.projections.0.delay_steps", -1],
  ["topology.projections.0.delay_mode", ""], ["topology.projections.0.n_synapses", -1],
  ["topology.projections.0.autapses_removed", -1], ["topology.projections.0.csr_sha256", 7],
  ["topology.projections.0.csr_sha256", "bad"], ["topology.projections.0.rule", "unknown"],
  ["topology.projections.0.source", "absent"], ["topology.projections.0.target", "absent"],
  ["topology.projections.0.indptr", null], ["topology.projections.0.indices", null],
  ["topology.projections.0.indptr", []], ["topology.projections.0.indptr.0", 1],
  ["topology.projections.0.indptr.2", 2], ["topology.projections.0.indices", []],
  ["topology.projections.0.indptr.1", 2], ["topology.projections.0.indices.0", 1],
  ["topology.projections.1.id", "ei"], ["topology.n_synapses", 4],
  ["graph_summary.n_populations", 4], ["graph_summary.n_projections", 3], ["graph_summary.n_neurons", 5],
  ["graph_summary.n_synapses", 4], ["graph_summary.n_excitatory", 4], ["graph_summary.n_inhibitory", 2],
] as const)("refuses a byte-sealed but invalid scientific field %s", async (path, value) => {
  const native = await sample();
  const artifact: unknown = JSON.parse(native.text);
  const keys = path.split(".");
  let parent = container(artifact);
  for (const key of keys.slice(0, -1)) parent = container(parent[key]);
  const key = keys.at(-1);
  if (key === undefined) throw new Error("Empty contract field path");
  parent[key] = value;
  const bytes = new TextEncoder().encode(JSON.stringify(artifact));
  await expect(parseGraphJobResult(bytes, await rebound(native.job, bytes))).rejects.toThrow("scientific contract");
});

it("rejects receipt count/identity changes independently of valid artifact bytes", async () => {
  const native = await sample();
  if (native.job.result === null) throw new Error("Missing result");
  for (const field of ["n_steps", "n_total", "n_spikes"] as const) {
    const job = { ...native.job, result: { ...native.job.result, [field]: native.job.result[field] + 1 } };
    await expect(parseGraphJobResult(native.bytes, job)).rejects.toThrow("scientific contract");
  }
  await expect(parseGraphJobResult(native.bytes, { ...native.job,
    result: { ...native.job.result, graph_sha256: "0".repeat(64) } })).rejects.toThrow("scientific contract");
});

it("accepts the unsigned 32-bit seed boundary and recognises explicitly omitted connectivity", async () => {
  const native = await sample();
  const boundary = new TextEncoder().encode(native.text.replaceAll('"seed":123', '"seed":4294967295'));
  const decoded = await parseGraphJobResult(boundary, await rebound(native.job, boundary));
  expect(decoded.topology.projections[0]?.seed).toBe(4294967295);
  const oversized = new TextEncoder().encode(native.text.replaceAll('"seed":123', '"seed":9007199254740993'));
  await expect(parseGraphJobResult(oversized, await rebound(native.job, oversized))).rejects.toThrow("scientific contract");
  const value = container(JSON.parse(native.text));
  const topology = container(value.topology);
  topology.connectivity_included = false;
  topology.connectivity_omitted_reason = "Connectivity budget; digests only";
  const projections = topology.projections;
  if (!Array.isArray(projections)) throw new Error("Missing projections");
  for (const item of projections) { delete container(item).indptr; delete container(item).indices; }
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  expect((await parseGraphJobResult(bytes, await rebound(native.job, bytes))).topology.connectivity_included).toBe(false);
  topology.connectivity_omitted_reason = null;
  const invalid = new TextEncoder().encode(JSON.stringify(value));
  await expect(parseGraphJobResult(invalid, await rebound(native.job, invalid))).rejects.toThrow("scientific contract");
  topology.connectivity_omitted_reason = "Digests only";
  container(projections[0]).indices = [];
  const invented = new TextEncoder().encode(JSON.stringify(value));
  await expect(parseGraphJobResult(invented, await rebound(native.job, invented))).rejects.toThrow("scientific contract");
});

it("refuses invalid UTF-8, broken JSON and non-finite scientific input metadata after correct byte hashing", async () => {
  const native = await sample();
  for (const bytes of [new Uint8Array([0xff]), new TextEncoder().encode("{broken"),
    new TextEncoder().encode(native.text.replace('"tau":10.0', '"tau":1e400'))]) {
    await expect(parseGraphJobResult(bytes, await rebound(native.job, bytes))).rejects.toThrow();
  }
});
