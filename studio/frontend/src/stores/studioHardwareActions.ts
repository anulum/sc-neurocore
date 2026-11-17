// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio hardware compilation and synthesis actions

import {
  fetchStudioJobs,
  fetchStudioOperatorStatus,
  fetchSynthTools,
  runSynthesis as apiRunSynthesis,
  runSynthesisTerminal as apiRunSynthesisTerminal,
  runMultiTargetSynthesis,
  fetchSynthEstimate,
} from "../api/client";
import {
  multiTargetSynthesisRunCompletedState,
  synthesisErrorMessageState,
  synthesisErrorState,
  synthesisEstimateLoadedState,
  synthesisRunCompletedState,
  synthesisTargetState,
  synthesisToolStatusLoadedState,
} from "../synthesisStoreState";
import { runStoreCompile } from "./studioCompile";
import { runStoreSynthesis } from "./studioSynthesis";
import { studioStageSet } from "./studioStageFailure";
import type { StudioState, StudioStoreActions } from "./studioTypes";

/**
 * The hardware actions: compiling the model to RTL, co-simulating that RTL against the model, and
 * synthesising it: choosing the target, running single- and multi-target synthesis, estimating
 * resources from the built IR and checking the synthesis tools.
 */
export type StudioHardwareActions = Pick<StudioStoreActions,
  "runCompile" | "runCosim" | "setSynthTarget" | "runSynthesis" | "runMultiTargetSynthesis" | "runSynthEstimate" | "checkSynthTools">;

/**
 * Bind these actions to the Studio store.
 *
 * The hardware actions: compiling the model to RTL, co-simulating that RTL against the model, and
 * synthesising it: choosing the target, running single- and multi-target synthesis, estimating
 * resources from the built IR and checking the synthesis tools. A selected catalogue model goes to
 * the digest-bound place-and-route terminal only when its RTL has current bit-exact co-simulation
 * parity; ODE equations go to plain synthesis.
 *
 * @param set - Writes a patch into the store.
 * @param get - Reads the store as it stands.
 * @returns The actions.
 */
export function createStudioHardwareActions(
  set: (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
  get: () => StudioState,
): StudioHardwareActions {
  return {
  runCompile: () => runStoreCompile("compile", get, studioStageSet("compile", get, set)),

  runCosim: () => runStoreCompile("cosim", get, studioStageSet("cosim", get, set)),

  setSynthTarget: (t) => { set(synthesisTargetState(t)); },

  runSynthesis: () => runStoreSynthesis(get, studioStageSet("synthesise", get, set), async (s) => {
      const verilog = s.svSource || s.verilogSrc;
      let resultArtifactPath = "synthesis/result.json";
      let synthResult: NonNullable<StudioState["synthResult"]>;
      if (s.sourceMode === "model") {
        if (s.compileTraceability === null || s.cosimResult === null) {
          throw new Error("Compile and bit-exact co-simulate the selected model before routing.");
        }
        if (!s.cosimResult.bit_exact
          || s.cosimResult.rtl.source_sha256 !== s.compileTraceability.output.rtl_sha256) {
          throw new Error("Selected RTL does not have current bit-exact co-simulation parity.");
        }
        if (!new Set(["ice40", "ecp5"]).has(s.synthTarget)) {
          throw new Error(`Target ${s.synthTarget} has no selected-RTL place-and-route terminal.`);
        }
        const terminal = await apiRunSynthesisTerminal(
          verilog,
          s.synthTarget,
          s.compileTraceability,
          s.cosimResult,
        );
        synthResult = { ...terminal.synthesis, silicon_terminal: terminal,
          studio_job_receipt: terminal.studio_job_receipt };
        resultArtifactPath = "synthesis/terminal-result.json";
      } else {
        synthResult = await apiRunSynthesis(verilog, s.synthTarget);
      }
      const [operatorStatus, jobList] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
      ]);
      return synthesisRunCompletedState(
        synthResult,
        operatorStatus,
        jobList,
        resultArtifactPath,
      );
  }),

  runMultiTargetSynthesis: () => runStoreSynthesis(get, studioStageSet("synthesise", get, set), async (s) => {
    if (s.sourceMode === "model") {
      throw new Error(
        "Selected models use the digest-bound single-target synthesis/PnR terminal.",
      );
    }
      const verilog = s.svSource || s.verilogSrc;
      const multiTargetResult = await runMultiTargetSynthesis(verilog);
      const [operatorStatus, jobList] = await Promise.all([
        fetchStudioOperatorStatus(),
        fetchStudioJobs(),
      ]);
      return multiTargetSynthesisRunCompletedState(multiTargetResult, operatorStatus, jobList);
  }),

  runSynthEstimate: async () => {
    const s = get();
    const irOps = s.irText ? s.irText.split("\n").filter((l) => l.trim().startsWith("%")).length : 0;
    if (irOps < 1) {
      set(synthesisErrorMessageState("Build IR first to estimate resources"));
      return;
    }
    try {
      const synthEstimate = await fetchSynthEstimate(irOps, s.synthTarget);
      set(synthesisEstimateLoadedState(synthEstimate));
    } catch (e) { set(synthesisErrorState(e, "Synthesis estimate failed")); }
  },

  checkSynthTools: async () => {
    try {
      const toolsAvailable = await fetchSynthTools();
      set(synthesisToolStatusLoadedState(toolsAvailable));
    } catch { /* tools check is non-critical */ }
  },
  };
}
