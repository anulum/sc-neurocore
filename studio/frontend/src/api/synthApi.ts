// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: synth endpoints.
import { post, get } from "./http";
import type {
  SynthToolInfo,
  SynthResult,
  SynthEstimate,
  MultiTargetResult,
  PnRResult,
  CompileTraceability,
  ModelCosimReport,
  SiliconTerminalResult,
} from "./types";

/**
 * Report which synthesis tools the server can actually reach.
 *
 * A tool that is absent is reported as absent rather than assumed, so a
 * synthesis that cannot run says so before it is started.
 *
 * @returns Each tool by name, with whether it is available and its version.
 */
export const fetchSynthTools = () => get<Record<string, SynthToolInfo>>("/synth/tools-status");

/**
 * Synthesise RTL for one target.
 *
 * @param verilog - The RTL to synthesise.
 * @param target - The device family to synthesise for.
 * @returns Resource use, utilisation against the device, and the log excerpt.
 */
export const runSynthesis = (verilog: string, target: string) =>
  post<SynthResult>("/synth/run", { verilog, target });

/**
 * Run the terminal step: synthesise, place and route, and seal the chain.
 *
 * The traceability and the co-simulation parity are sent with it because the
 * result is evidence that a specific model became specific silicon; without
 * them the artefacts would say what was built but not what it came from.
 *
 * @param verilog - The RTL to take to silicon.
 * @param target - The device family.
 * @param compileTraceability - Where the RTL came from.
 * @param cosimParity - The report that the RTL matched the model.
 * @returns The sealed terminal result, artefacts and source chain included.
 */
export const runSynthesisTerminal = (
  verilog: string,
  target: string,
  compileTraceability: CompileTraceability,
  cosimParity: ModelCosimReport,
) => post<SiliconTerminalResult>("/synth/terminal", {
  compile_traceability: compileTraceability,
  cosim_parity: cosimParity,
  target,
  verilog,
});

/**
 * Synthesise the same RTL for every supported target, for comparison.
 *
 * @param verilog - The RTL to synthesise.
 * @returns One result per target, and the provenance of each device's figures.
 */
export const runMultiTargetSynthesis = (verilog: string) =>
  post<MultiTargetResult>("/synth/multi-target", { verilog });

/**
 * Estimate resource use from an IR operation count, without synthesising.
 *
 * An estimate, explicitly: it is for sizing a design before committing to a
 * synthesis run, and it is not evidence of what the device will hold.
 *
 * @param irOpCount - Operations in the IR.
 * @param target - The device family to estimate against.
 * @returns The estimated resource use.
 */
export const fetchSynthEstimate = (irOpCount: number, target: string) =>
  post<SynthEstimate>("/synth/estimate", { ir_op_count: irOpCount, target });

/**
 * Place and route a netlist the server already holds.
 *
 * @param jsonPath - Server-side path of the netlist to route.
 * @param target - The device family.
 * @returns The routed design, its critical path and its maximum frequency.
 */
export const runPnR = (jsonPath: string, target: string) =>
  post<PnRResult>("/synth/pnr", { json_path: jsonPath, target });
