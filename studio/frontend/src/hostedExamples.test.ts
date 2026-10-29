// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — editable scientific example format contracts

import { expect, it } from "vitest";
import { HOST_OPERATIONS } from "./hostApiTypes";
import { hostedExample, HOST_OPERATION_LABELS } from "./hostedExamples";

it.each(HOST_OPERATIONS)("produces an editable scientific request and domain label for %s", (operation) => {
  const json = hostedExample(operation, "");
  const parsed: unknown = JSON.parse(json);
  expect(parsed).toBeTypeOf("object"); expect(Array.isArray(parsed)).toBe(false);
  expect(HOST_OPERATION_LABELS[operation]).not.toBe("");
  expect(parsed).not.toHaveProperty("task_path"); expect(parsed).not.toHaveProperty("action");
});

it.each(["ir.verify", "ir.replay", "ir.emit-sv"] as const)(
  "preserves arbitrary graph text exactly in the %s scientific document", (operation) => {
    const graph = 'graph "quoted"\n18446744073709551615\nž';
    const parsed: unknown = JSON.parse(hostedExample(operation, graph));
    expect(parsed).toEqual(operation === "ir.emit-sv"
      ? { ir_text: graph, source_policy: "require-bound" } : { ir_text: graph });
  },
);

it("keeps the authored example's ordered source and maximum random keys lossless", () => {
  const parsed: unknown = JSON.parse(hostedExample("ir.build", ""));
  expect(parsed).toMatchObject({ schema_version: "studio.equation-ir-request.v1", model: {
    equations: [{ name: "z", expression: "(I-z+a)/tau" }, { name: "a", expression: "z-a" }],
    initial_state: [{ name: "z", value: 3 }, { name: "a", value: 1 }],
    rng_seed: "18446744073709551615", random_stream: 4294967295, method: "gauss_seidel",
  } });
});
