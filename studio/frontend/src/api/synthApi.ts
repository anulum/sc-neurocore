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

export const fetchSynthTools = () => get<Record<string, SynthToolInfo>>("/synth/tools-status");

export const runSynthesis = (verilog: string, target: string) =>
  post<SynthResult>("/synth/run", { verilog, target });

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

export const runMultiTargetSynthesis = (verilog: string) =>
  post<MultiTargetResult>("/synth/multi-target", { verilog });

export const fetchSynthEstimate = (irOpCount: number, target: string) =>
  post<SynthEstimate>("/synth/estimate", { ir_op_count: irOpCount, target });

export const runPnR = (jsonPath: string, target: string) =>
  post<PnRResult>("/synth/pnr", { json_path: jsonPath, target });
