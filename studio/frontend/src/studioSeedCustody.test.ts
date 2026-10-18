// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Public store actions preserve seed and restore custody

import { afterEach, expect, it, vi } from "vitest";
import { useStudioStore } from "./stores/studio";

const initial = useStudioStore.getState();
afterEach(() => { useStudioStore.setState(initial, true); });

it.each(["model", "ode"] as const)("binds exact seed edits in %s mode without altering inactive inputs", (sourceMode) => {
  const autoSimulate = vi.fn<() => void>();
  const modelParams = { seed: 9007199254740993n, rate_hz: 100 };
  useStudioStore.setState({ sourceMode, modelParams, autoSimulate });
  useStudioStore.getState().setSeed(9223372036854775807n);
  const state = useStudioStore.getState();
  expect(state.seed).toBe(9223372036854775807n);
  expect(state.modelParams).toEqual(sourceMode === "model" ? { rate_hz: 100 } : modelParams);
  expect(modelParams.seed).toBe(9007199254740993n);
  expect(autoSimulate).toHaveBeenCalledOnce();
});

it("restores a valid full-domain local session through the public action before scheduling its run", () => {
  const runSimulation = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
  useStudioStore.setState({ runSimulation, savedSessions: [{ name: "exact", state: {
    sourceMode: "ode", seed: 9223372036854775807n, modelParams: { seed: 9007199254740993n },
  } }] });
  useStudioStore.getState().loadSession("exact");
  expect(useStudioStore.getState().seed).toBe(9223372036854775807n);
  expect(useStudioStore.getState().modelParams.seed).toBe(9007199254740993n);
  expect(runSimulation).toHaveBeenCalledOnce();
});

it("refuses a malformed explicit local seed before replacing or running the editor", () => {
  const runSimulation = vi.fn<() => Promise<void>>().mockResolvedValue(undefined);
  useStudioStore.setState({ sourceMode: "ode", seed: 42, runSimulation,
    savedSessions: [{ name: "invalid", state: { sourceMode: "model", seed: 1.5 } }] });
  useStudioStore.getState().loadSession("invalid");
  expect(useStudioStore.getState().seed).toBe(42);
  expect(useStudioStore.getState().sourceMode).toBe("ode");
  expect(useStudioStore.getState().error).toContain("Saved simulation seed");
  expect(runSimulation).not.toHaveBeenCalled();
});
