// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio compiler store state helpers

/**
 * The compiler's state, one patch per stage of the toolchain.
 *
 * The two `Invalidated` transitions are the interesting ones: changing the
 * compile configuration discards the RTL that came from the old one, and
 * changing anything the co-simulation depended on discards its parity report.
 * Keeping either would leave the panel showing evidence for a design that no
 * longer exists.
 */
import type {
  CompileResponse,
  CompileTraceability,
  IRBuildResponse,
  SVDirectResponse,
  SVEmitResponse,
  ModelCosimReport,
} from "./api/client";

/** A message to show without claiming the compile ended. */
export interface CompilerErrorStatePatch {
  error: string;
}

/** A compile has begun, on the tab that will show it. */
export interface CompilerRunStartStatePatch {
  activeTab: "ir" | "verilog";
  error: null;
  isSimulating: true;
}

/** A compile failed, with the message to show. */
export interface CompilerFailureStatePatch {
  error: string;
  isSimulating: false;
}

/** Generated RTL arrived, with its traceability. */
export interface CompilerVerilogLoadedStatePatch {
  compileTraceability: CompileTraceability;
  isSimulating: false;
  verilogSrc: string;
}

/** A co-simulation parity report arrived. */
export interface CompilerCosimLoadedStatePatch {
  cosimResult: ModelCosimReport;
  isSimulating: false;
}

/** The configuration changed, so the RTL it produced no longer applies. */
export interface CompilerConfigurationInvalidatedStatePatch {
  svSource: "";
  irText: "";
  irErrors: string[];
  compileEvidenceBundle: null;
  compileEvidenceBundleError: null;
  compileTraceability: null;
  cosimResult: null;
  latestMultiTargetSynthesisJobId: null;
  latestSynthesisJobId: null;
  multiTargetResult: null;
  synthEstimate: null;
  synthResult: null;
  synthesisEvidenceBundle: null;
  synthesisEvidenceBundleError: null;
  verilogSrc: "";
}

/** Something the parity report depended on changed, so it no longer applies. */
export interface CompilerCosimInvalidatedStatePatch {
  cosimResult: null;
  synthResult?: null;
  multiTargetResult?: null;
  latestSynthesisJobId?: null;
  latestMultiTargetSynthesisJobId?: null;
}

/** A built intermediate representation arrived. */
export interface CompilerIRLoadedStatePatch {
  irErrors: string[];
  irText: string;
  isSimulating: false;
}

/** SystemVerilog emitted from the IR arrived. */
export interface CompilerSVLoadedStatePatch {
  svSource: string;
}

/** SystemVerilog emitted straight from the system arrived, with its IR. */
export interface CompilerSVDirectLoadedStatePatch {
  compileTraceability: CompileTraceability;
  irText: string;
  isSimulating: false;
  svSource: string;
}

/**
 * Show a message without claiming the compile ended.
 *
 * @param error - The message.
 * @returns The patch.
 */
export function compilerErrorState(error: string): CompilerErrorStatePatch {
  return { error };
}

/**
 * Mark a compile as begun and move to the tab that shows it.
 *
 * @param activeTab - The tab to show.
 * @returns The patch.
 */
export function compilerRunStartState(
  activeTab: CompilerRunStartStatePatch["activeTab"],
): CompilerRunStartStatePatch {
  return { activeTab, error: null, isSimulating: true };
}

/**
 * Report a failed compile.
 *
 * @param error - Whatever was thrown or rejected.
 * @returns The patch.
 */
export function compilerFailureState(error: unknown): CompilerFailureStatePatch {
  return {
    error: error instanceof Error && error.message.length > 0 ? error.message : String(error),
    isSimulating: false,
  };
}

/**
 * Take generated RTL into the store.
 *
 * @param response - The compile response.
 * @returns The patch.
 */
export function compilerVerilogLoadedState(
  response: CompileResponse,
): CompilerVerilogLoadedStatePatch {
  return {
    compileTraceability: response.compile_traceability,
    isSimulating: false,
    verilogSrc: response.verilog,
  };
}

/**
 * Take a co-simulation parity report into the store.
 *
 * @param response - The report.
 * @returns The patch.
 */
export function compilerCosimLoadedState(
  response: ModelCosimReport,
): CompilerCosimLoadedStatePatch {
  return { cosimResult: response, isSimulating: false };
}

/**
 * Discard RTL that came from a configuration no longer in force.
 *
 * @returns The patch.
 */
export function compilerConfigurationInvalidatedState(): CompilerConfigurationInvalidatedStatePatch {
  return {
    svSource: "",
    irText: "",
    irErrors: [],
    compileEvidenceBundle: null,
    compileEvidenceBundleError: null,
    compileTraceability: null,
    cosimResult: null,
    latestMultiTargetSynthesisJobId: null,
    latestSynthesisJobId: null,
    multiTargetResult: null,
    synthEstimate: null,
    synthResult: null,
    synthesisEvidenceBundle: null,
    synthesisEvidenceBundleError: null,
    verilogSrc: "",
  };
}

/**
 * Discard a parity report whose premises have changed.
 *
 * Model terminal synthesis depends on this report; withdraw its qualification
 * and export job handles too. Keep compiled RTL and historical evidence bundles.
 * ODE synthesis has no dependency on catalogue-model parity.
 *
 * @param modelMode - Whether synthesis depends on catalogue-model parity.
 * @returns The patch.
 */
export function compilerCosimInvalidatedState(modelMode: boolean): CompilerCosimInvalidatedStatePatch {
  return { cosimResult: null, ...(modelMode ? {
    synthResult: null, multiTargetResult: null,
    latestSynthesisJobId: null, latestMultiTargetSynthesisJobId: null,
  } : {}) };
}

/**
 * Take a built IR into the store.
 *
 * @param response - The build response.
 * @returns The patch.
 */
export function compilerIRLoadedState(
  response: IRBuildResponse,
): CompilerIRLoadedStatePatch {
  return {
    irErrors: response.errors,
    irText: response.ir_text,
    isSimulating: false,
  };
}

/**
 * Take emitted SystemVerilog into the store.
 *
 * @param response - The emit response.
 * @returns The patch.
 */
export function compilerSVLoadedState(response: SVEmitResponse): CompilerSVLoadedStatePatch {
  return { svSource: response.systemverilog };
}

/**
 * Take directly-emitted SystemVerilog into the store, with its IR.
 *
 * @param response - The emit response.
 * @returns The patch.
 */
export function compilerSVDirectLoadedState(
  response: SVDirectResponse,
): CompilerSVDirectLoadedStatePatch {
  return {
    compileTraceability: response.compile_traceability,
    irText: response.ir_repr,
    isSimulating: false,
    svSource: response.verilog,
  };
}
