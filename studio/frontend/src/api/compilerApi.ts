// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: compiler endpoints.
import { post } from "./http";
import type {
  PrecisionResponse,
  CompileResponse,
  ModelCompileRequest,
  ModelCosimReport,
  ModelCosimRequest,
  IRBuildResponse,
  IRVerifyResponse,
  SVEmitResponse,
  SVDirectResponse,
} from "./types";

/**
 * Compile a custom ODE system to Verilog.
 *
 * @param req - The system and the compilation settings.
 * @returns The generated RTL and the report of how it was produced.
 */
export const compileVerilog = (req: Record<string, unknown>) => post<CompileResponse>("/compile", req);

/**
 * Compile a catalogue model to Verilog.
 *
 * @param req - The model, the fixed-point format and the target.
 * @returns The generated RTL and the report of how it was produced.
 */
export const compileModelVerilog = (req: ModelCompileRequest) => (
  post<CompileResponse>("/models/compile", req)
);

/**
 * Co-simulate a model's RTL against its reference implementation.
 *
 * This is the step that says whether the hardware agrees with the model, so
 * its report is evidence rather than a convenience.
 *
 * @param req - The model, the RTL, and the trace to compare over.
 * @returns The comparison and its verdict.
 */
export const cosimModelVerilog = (req: ModelCosimRequest) => (
  post<ModelCosimReport>("/models/cosim", req)
);

/**
 * Build the intermediate representation the emitters work from.
 *
 * @param req - The system to lower.
 * @returns The IR text and the report of how it was built.
 */
export const buildIR = (req: Record<string, unknown>) => post<IRBuildResponse>("/ir/build", req);

/**
 * Check an IR document against the invariants the emitters rely on.
 *
 * @param irText - The IR to check.
 * @returns What it violates, or that it violates nothing.
 */
export const verifyIR = (irText: string) => post<IRVerifyResponse>("/ir/verify", { ir_text: irText });

/**
 * Emit SystemVerilog from an IR document.
 *
 * @param irText - The IR to emit from.
 * @returns The emitted SystemVerilog and its report.
 */
export const emitSV = (irText: string) => post<SVEmitResponse>("/ir/emit-sv", { ir_text: irText });

/**
 * Emit SystemVerilog straight from a system, without a separate IR round trip.
 *
 * @param req - The system and the emission settings.
 * @returns The emitted SystemVerilog and its report.
 */
export const emitSVDirect = (req: Record<string, unknown>) => post<SVDirectResponse>("/ir/emit-sv-direct", req);

/**
 * Co-simulate at IR level and return the trace-by-trace difference.
 *
 * @param req - The IR, the experiment, and the arithmetic to compare under.
 * @returns Both traces and the error between them.
 */
export const fetchCosimDetail = (req: Record<string, unknown>) => post<PrecisionResponse>("/ir/cosim", req);
