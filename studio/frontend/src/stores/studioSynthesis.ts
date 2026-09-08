// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Synthesis lifecycle and source ownership

import { canonicalSealText } from "../evidenceSeal";
import { synthesisFailureState, synthesisRunStartState } from "../synthesisStoreState";
import type { StudioState } from "./studioTypes";

/**
 * Identify generated source, target and model-terminal prerequisites.
 *
 * @param state - Current synthesis inputs.
 * @returns Local identity; not a backend job ID or receipt digest.
 */
function synthesisKey(state: StudioState): string {
  return canonicalSealText({ source: state.sourceMode,
    verilog: state.svSource || state.verilogSrc, target: state.synthTarget,
    compile: state.sourceMode === "model" ? state.compileTraceability : null,
    parity: state.sourceMode === "model" ? state.cosimResult : null });
}

/**
 * Run synthesis and its operator refresh as one owned asynchronous request.
 *
 * Previous results and export handles no longer attest a rerun. Both success
 * and failure must still match current inputs; duplicate starts change nothing.
 * The request callback retains existing single/multi-target validation and API.
 *
 * @param get - Read current state across awaits.
 * @param set - Apply state patches without replacing unrelated work.
 * @param request - Execute the captured synthesis and refresh operation.
 */
export async function runStoreSynthesis(
  get: () => StudioState, set: (patch: Partial<StudioState>) => void,
  request: (state: StudioState) => Promise<Partial<StudioState>>,
): Promise<void> {
  const state = get();
  if (state.isSimulating) return;
  set({ ...synthesisRunStartState(), synthResult: null,
    latestMultiTargetSynthesisJobId: null });
  let key: string | null = null;
  try {
    key = synthesisKey(state);
    if (!state.svSource && !state.verilogSrc) throw new Error("Generate Verilog first");
    const patch = await request(state);
    if (synthesisKey(get()) !== key) { set({ isSimulating: false }); return; }
    set(patch);
  } catch (error: unknown) {
    let failure = error;
    try {
      if (key !== null && synthesisKey(get()) !== key) {
        set({ isSimulating: false }); return;
      }
    } catch (currentInputError: unknown) { failure = currentInputError; }
    set(synthesisFailureState(failure));
  }
}
