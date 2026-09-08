// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compile and co-simulation request ownership

import { buildIR, emitSV, emitSVDirect, compileModelVerilog, compileVerilog, cosimModelVerilog } from "../api/client";
import { canonicalSealText } from "../evidenceSeal";
import { modelCompileRequest, modelCosimRequest } from "../modelCompileConfig";
import { compilerConfigurationInvalidatedState, compilerCosimInvalidatedState,
  compilerCosimLoadedState, compilerFailureState, compilerRunStartState,
  compilerVerilogLoadedState, compilerIRLoadedState, compilerSVLoadedState,
  compilerSVDirectLoadedState } from "../compilerStoreState";
import type { StudioState } from "./studioTypes";

/** Compiler entry points sharing exclusive request ownership. */
type Operation = "compile" | "cosim" | "ir" | "sv";

/**
 * Resolve a request without sending it.
 *
 * @param state - Captured compiler inputs.
 * @param operation - Compile or parity operation.
 * @returns Request identity and deferred execution.
 */
function plan(state: StudioState, operation: Operation) {
  if (operation === "ir" || operation === "sv") {
    if (state.sourceMode !== "ode") throw new Error("IR/SV generation requires ODE mode");
    const system = { equations: state.equations, threshold: state.threshold || null,
      reset: state.reset || null, params: state.odeParams };
    if (operation === "sv") {
      const request = { ...system, init: state.odeInit };
      return { key: canonicalSealText({ operation, request }),
        execute: async () => compilerSVDirectLoadedState(await emitSVDirect(request)) };
    }
    const request = { ...system, dt: state.dt };
    return { key: canonicalSealText({ operation, request }), execute: async () => {
      const ir = await buildIR(request);
      const patch = compilerIRLoadedState(ir);
      if (ir.errors.length > 0) return patch;
      return { ...patch, ...compilerSVLoadedState(await emitSV(ir.ir_text)) };
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
  set({ ...(operation === "cosim" ? compilerCosimInvalidatedState() : compilerConfigurationInvalidatedState()),
    ...compilerRunStartState(operation === "ir" || operation === "sv" ? "ir" : "verilog") });
  let requestedKey: string | null = null;
  try {
    const request = plan(state, operation);
    requestedKey = request.key;
    const patch = await request.execute();
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
