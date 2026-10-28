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
