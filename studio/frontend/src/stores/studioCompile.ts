// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compile and co-simulation request ownership

import { buildIR, emitSV, verifyIR, replayIR, compileModelVerilog, compileVerilog, cosimModelVerilog } from "../api/client";
import { canonicalSealText } from "../evidenceSeal";
import { equationIrRequest } from "../equationIrConfig";
import { modelCompileRequest, modelCosimRequest } from "../modelCompileConfig";
import { compilerConfigurationInvalidatedState, compilerCosimInvalidatedState,
  compilerCosimLoadedState, compilerFailureState, compilerRunStartState,
  compilerVerilogLoadedState, compilerIRLoadedState, compilerSVLoadedState } from "../compilerStoreState";
import type { StudioState } from "./studioTypes";

/** Compiler entry points sharing exclusive request ownership. */
type Operation = "compile" | "cosim" | "ir" | "sv" | "verify-ir" | "replay-ir";

/** Captured request identity and execution with an ownership check between native jobs. */
interface CompilePlan {
  key: string;
  execute: (isCurrent: () => boolean) => Promise<Partial<StudioState>>;
}

/**
 * Resolve a request without sending it.
 *
 * @param state - Captured compiler inputs.
 * @param operation - Compile or parity operation.
 * @returns Request identity and deferred execution.
 */
function plan(state: StudioState, operation: Operation): CompilePlan {
  if (operation === "ir" || operation === "sv" || operation === "verify-ir" || operation === "replay-ir") {
    if (state.sourceMode !== "ode") throw new Error("IR/SV generation requires ODE mode");
    const request = equationIrRequest(state);
    const sourceKey = canonicalSealText(request);
    if (operation === "verify-ir" || operation === "replay-ir") {
      if (!state.irText || state.irRequestKey !== sourceKey) {
        throw new Error("Build IR for the current equation inputs before inspecting it.");
      }
      return { key: canonicalSealText({ operation, request, ir_text: state.irText }),
        execute: async () => compilerIRLoadedState(
          await (operation === "verify-ir" ? verifyIR(state.irText) : replayIR(state.irText)), sourceKey,
        ) };
    }
    return { key: canonicalSealText({ operation, request }), execute: async (isCurrent) => {
      if (operation === "sv" && state.irText && state.irRequestKey === sourceKey) {
        const emitted = await emitSV(state.irText);
        return { ...compilerIRLoadedState(emitted, sourceKey), ...compilerSVLoadedState(emitted) };
      }
      const ir = await buildIR(request);
      if (operation === "ir") return compilerIRLoadedState(ir, sourceKey);
      if (!isCurrent()) return { isSimulating: false };
      const emitted = await emitSV(ir.ir_text);
      return { ...compilerIRLoadedState(emitted, sourceKey), ...compilerSVLoadedState(emitted) };
    } };
  }
  const input = { dt: state.dt, integrator: state.modelIntegrator,
    modelDetail: state.modelDetail, modelParams: state.modelParams,
    qFormat: state.modelQFormat, selectedModelName: state.selectedModelName };
  if (operation === "cosim") {
    if (state.sourceMode !== "model") throw new Error("Bit-exact selected-model co-simulation requires catalogue model mode.");
    const request = modelCosimRequest(input, { current: state.current });
    return { key: canonicalSealText({ operation, request }),
      execute: async () => compilerCosimLoadedState(await cosimModelVerilog(request)) };
  }
  if (state.sourceMode === "model") {
    const request = modelCompileRequest(input);
    return { key: canonicalSealText({ operation, source: "model", request }),
      execute: async () => compilerVerilogLoadedState(await compileModelVerilog(request)) };
  }
  const request = { equations: state.equations, threshold: state.threshold,
    reset: state.reset, params: state.odeParams, init: state.odeInit };
  return { key: canonicalSealText({ operation, source: "ode", request }),
    execute: async () => compilerVerilogLoadedState(await compileVerilog(request)) };
}

/**
 * Publish only an outcome for the current resolved compiler request.
 *
 * Reruns withdraw prior completion. Busy invocations leave state untouched;
 * obsolete outcomes cannot restore invalidated evidence. Identity uses actual
 * request fields, not unrelated simulation settings or a new server digest.
 *
 * @param operation - Compiler surface to execute.
 * @param get - Read live state when a response arrives.
 * @param set - Apply patches without replacing unrelated state.
 */
export async function runStoreCompile(
  operation: Operation, get: () => StudioState,
  set: (patch: Partial<StudioState>) => void,
): Promise<void> {
  const state = get();
  if (state.isSimulating) return;
  const inspecting = operation === "verify-ir" || operation === "replay-ir";
  set({ ...(operation === "cosim" ? compilerCosimInvalidatedState(state.sourceMode === "model") : compilerConfigurationInvalidatedState()),
    ...(inspecting ? { irText: state.irText, irRequestKey: state.irRequestKey } : {}),
    ...compilerRunStartState(operation === "compile" || operation === "cosim" ? "verilog" : "ir") });
  let requestedKey: string | null = null;
  try {
    const request = plan(state, operation);
    requestedKey = request.key;
    const patch = await request.execute(
      () => get().sourceMode === state.sourceMode && plan(get(), operation).key === request.key,
    );
    if (get().sourceMode !== state.sourceMode || plan(get(), operation).key !== requestedKey) {
      set({ isSimulating: false }); return;
    }
    set(patch);
  } catch (error: unknown) {
    let failure = error;
    try {
      if (requestedKey !== null && (get().sourceMode !== state.sourceMode || plan(get(), operation).key !== requestedKey)) {
        set({ isSimulating: false }); return;
      }
    } catch (currentInputError: unknown) { failure = currentInputError; }
    set(compilerFailureState(failure));
  }
}
