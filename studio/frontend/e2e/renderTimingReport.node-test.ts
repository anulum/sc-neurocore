// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — render timing artefact: statistics and schema

/** Exercise the render timing statistics and artefact schema on fixed vectors, with native Node coverage. */
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  MIN_SAMPLES,
  RENDER_TIMING_SCHEMA,
  RenderTimingError,
  buildArtefact,
  commitLabel,
  isolationLabel,
  parseArtefact,
  percentile,
  summarise,
} from "./renderTimingReport.ts";
import type { RenderTimingEnvironment } from "./renderTimingReport.ts";

const TEN = [12, 3, 7, 9, 1, 15, 4, 8, 10, 6];

const ENVIRONMENT: RenderTimingEnvironment = {
  host: "workstation",
  cpuModel: "Example CPU",
  loadBefore: [1.5, 1.25, 1],
  loadAfter: [2, 1.5, 1.25],
  browser: "chromium 140.0",
  commit: "0123456789abcdef0123456789abcdef01234567",
  isolation: "functional-non-isolated",
};

await test("the nearest-rank percentile of fixed vectors", () => {
  const sorted = [...TEN].sort((a, b) => a - b);
  assert.equal(percentile(sorted, 50), 7);
  assert.equal(percentile(sorted, 95), 15);
  assert.equal(percentile(sorted, 100), 15);
  assert.equal(percentile(sorted, 10), 1);
  assert.equal(percentile([5], 50), 5);
  assert.equal(percentile([1, 2, 3, 4], 50), 2);
  assert.equal(percentile([1, 2, 3, 4], 75), 3);
  assert.equal(percentile([1, 2, 3, 4], 76), 4);
});

await test("a percentile outside (0, 100] or of no samples is refused", () => {
  for (const p of [0, -1, 101, Number.NaN]) {
    assert.throws(() => percentile([1, 2], p), { message: `the percentile must lie in (0, 100], got ${String(p)}` });
  }
  assert.throws(() => percentile([], 50), { message: "no samples to take a percentile of" });
});

await test("a measure keeps every sample in its order and adds p50 and p95", () => {
  assert.deepEqual(summarise({ name: "trace", workload: "one model, 1000 steps", samplesMs: TEN }), {
    name: "trace",
    workload: "one model, 1000 steps",
    samplesMs: TEN,
    count: 10,
    p50Ms: 7,
    p95Ms: 15,
  });
});

await test("fewer than ten samples are refused", () => {
  assert.equal(MIN_SAMPLES, 10);
  assert.throws(() => summarise({ name: "trace", workload: "one model", samplesMs: TEN.slice(0, 9) }), {
    message: "trace (one model) has 9 samples; a measure needs at least 10",
  });
});

await test("a sample that is not a finite, non-negative duration is refused", () => {
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, -0.5]) {
    assert.throws(() => summarise({ name: "view", workload: "catalogue", samplesMs: [...TEN.slice(0, 9), bad] }), {
      message: `view (catalogue) has a sample that is not a duration: ${String(bad)}`,
    });
  }
});

await test("the artefact carries its schema, time, environment and summaries, and survives JSON", () => {
  const artefact = buildArtefact([{ name: "trace", workload: "one model", samplesMs: TEN }], ENVIRONMENT, "2026-11-13T10:00:00Z");
  assert.equal(artefact.schema, RENDER_TIMING_SCHEMA);
  assert.equal(artefact.recordedAt, "2026-11-13T10:00:00Z");
  assert.deepEqual(artefact.environment, ENVIRONMENT);
  assert.equal(artefact.measures[0]?.p95Ms, 15);
  assert.deepEqual(parseArtefact(JSON.parse(JSON.stringify(artefact))), artefact);
});

await test("an artefact without a measure is refused", () => {
  assert.throws(() => buildArtefact([], ENVIRONMENT, "2026-11-13T10:00:00Z"), { message: "the artefact names no measure" });
});

await test("a damaged artefact is refused with its reason", () => {
  const good = JSON.parse(JSON.stringify(buildArtefact([{ name: "trace", workload: "one model", samplesMs: TEN }], ENVIRONMENT, "2026-11-13T10:00:00Z"))) as Record<string, unknown>;
  const cases: [unknown, string][] = [
    [null, "the artefact is not an object"],
    [[], "the artefact is not an object"],
    [{ ...good, schema: "other" }, `the artefact's schema is not ${RENDER_TIMING_SCHEMA}`],
    [{ ...good, recordedAt: 5 }, "the artefact's recordedAt is not text"],
    [{ ...good, environment: null }, "the artefact's environment is not an object"],
    [{ ...good, environment: { ...ENVIRONMENT, host: 3 } }, "the artefact's environment.host is not text"],
    [{ ...good, environment: { ...ENVIRONMENT, loadBefore: [1, "x", 2] } }, "the artefact's environment.loadBefore is not three numbers"],
    [{ ...good, environment: { ...ENVIRONMENT, loadAfter: [1] } }, "the artefact's environment.loadAfter is not three numbers"],
    [{ ...good, environment: { ...ENVIRONMENT, isolation: "fast" } }, "the artefact's environment.isolation is not a known label"],
    [{ ...good, measures: "x" }, "the artefact's measures are not a list"],
    [{ ...good, measures: [] }, "the artefact names no measure"],
    [{ ...good, measures: [null] }, "measure 0 is not an object"],
    [{ ...good, measures: [{ name: "trace", workload: "one model", samplesMs: "x" }] }, "measure 0 has no list of samples"],
    [{ ...good, measures: [{ name: 1, workload: "one model", samplesMs: TEN }] }, "measure 0 has no name and workload"],
    [{ ...good, measures: [{ name: "trace", workload: "one model", samplesMs: TEN, p50Ms: 99, p95Ms: 15, count: 10 }] }, "measure 0 (trace, one model) does not match its samples"],
  ];
  for (const [value, message] of cases) {
    assert.throws(() => parseArtefact(value), { message }, message);
  }
  assert.ok(new RenderTimingError("x") instanceof Error);
});

await test("a run is labelled isolated only on a reserved core set", () => {
  assert.equal(isolationLabel(undefined), "functional-non-isolated");
  assert.equal(isolationLabel(""), "functional-non-isolated");
  assert.equal(isolationLabel("  "), "functional-non-isolated");
  assert.equal(isolationLabel("2-5"), "isolated-reserved-cores");
});

await test("a checkout whose tracked files differ from its commit is not named by the commit alone", () => {
  const head = ENVIRONMENT.commit;
  assert.equal(commitLabel(head, ""), head);
  assert.equal(commitLabel(head, " \n"), head);
  assert.equal(commitLabel(head, "M  studio/frontend/src/renderTiming.ts\n"), `${head} with uncommitted changes`);
  assert.equal(commitLabel(head, " M README.md\nA  docs/new.md\n"), `${head} with uncommitted changes`);
});
