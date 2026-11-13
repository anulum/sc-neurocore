// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compiler export context follows native graph custody

import { afterEach, expect, it } from "vitest";
import { nativeIRBuildResult } from "./equationIrTestSupport";
import { studioBundleContextKey } from "./studioBundleContext";
import { useStudioStore } from "./stores/studio";

const initial = useStudioStore.getState();
afterEach(() => { useStudioStore.setState(initial, true); });

it("changes compile export ownership for a native graph before RTL exists", () => {
  const before = useStudioStore.getState();
  const result = nativeIRBuildResult();
  useStudioStore.setState({ irText: result.ir_text, irRequestKey: "authored request", irResult: result });
  const built = useStudioStore.getState();
  expect(built.svSource).toBe("");
  expect(studioBundleContextKey("compile", built)).not.toBe(studioBundleContextKey("compile", before));
  useStudioStore.getState().setEquationIrConfig(built.equationIrConfig);
  expect(studioBundleContextKey("compile", useStudioStore.getState()))
    .not.toBe(studioBundleContextKey("compile", built));
});

it.each([
  ["an emptied seed", { seed: undefined }],
  ["the all_to_all rule, which drops the probability", { probability: undefined, rule: "all_to_all" as const }],
])("keeps the project export's key computable after %s", (_change, updates) => {
  useStudioStore.setState({
    graphProjections: [{ delay: 0, id: "e1", probability: 0.2, rule: "random", seed: 7, source: "p1", target: "p2",
      weight: 8 }],
  });
  useStudioStore.getState().updateProjection("e1", updates);
  // The key seals the project's save state; a field left as undefined made it throw and refused the export.
  expect(() => studioBundleContextKey("project", useStudioStore.getState())).not.toThrow();
});

it("binds repeat compilation exports to their actual job receipt", () => {
  const result = nativeIRBuildResult();
  useStudioStore.setState({ irText: result.ir_text, irRequestKey: "authored request", irResult: result });
  const before = studioBundleContextKey("compile", useStudioStore.getState());
  useStudioStore.setState({ irResult: { ...result, studio_job_receipt: {
    ...result.studio_job_receipt, job_id: "sj_repeat_operation",
  } } });
  expect(studioBundleContextKey("compile", useStudioStore.getState())).not.toBe(before);
  expect(studioBundleContextKey("admin", useStudioStore.getState())).toBe("admin");
});
