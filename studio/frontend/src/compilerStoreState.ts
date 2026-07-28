// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio compiler store state helpers
import type {
  CompileResponse,
  CompileTraceability,
  IRBuildResponse,
  SVDirectResponse,
  SVEmitResponse,
  ModelCosimReport,
} from "./api/client";

export interface CompilerErrorStatePatch {
  error: string;
}

export interface CompilerRunStartStatePatch {
  activeTab: "ir" | "verilog";
  error: null;
  isSimulating: true;
}

export interface CompilerFailureStatePatch {
  error: string;
  isSimulating: false;
}

export interface CompilerVerilogLoadedStatePatch {
  compileTraceability: CompileTraceability;
  isSimulating: false;
  verilogSrc: string;
}

export interface CompilerCosimLoadedStatePatch {
  cosimResult: ModelCosimReport;
  isSimulating: false;
}

export interface CompilerConfigurationInvalidatedStatePatch {
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

export interface CompilerCosimInvalidatedStatePatch {
  cosimResult: null;
}

export interface CompilerIRLoadedStatePatch {
  irErrors: string[];
  irText: string;
  isSimulating: false;
}

export interface CompilerSVLoadedStatePatch {
  svSource: string;
}

export interface CompilerSVDirectLoadedStatePatch {
  compileTraceability: CompileTraceability;
  irText: string;
  isSimulating: false;
  svSource: string;
}

export function compilerErrorState(error: string): CompilerErrorStatePatch {
  return { error };
}

export function compilerRunStartState(
  activeTab: CompilerRunStartStatePatch["activeTab"],
): CompilerRunStartStatePatch {
  return { activeTab, error: null, isSimulating: true };
}

export function compilerFailureState(error: unknown): CompilerFailureStatePatch {
  return {
    error: error instanceof Error && error.message.length > 0 ? error.message : String(error),
    isSimulating: false,
  };
}

export function compilerVerilogLoadedState(
  response: CompileResponse,
): CompilerVerilogLoadedStatePatch {
  return {
    compileTraceability: response.compile_traceability,
    isSimulating: false,
    verilogSrc: response.verilog,
  };
}

export function compilerCosimLoadedState(
  response: ModelCosimReport,
): CompilerCosimLoadedStatePatch {
  return { cosimResult: response, isSimulating: false };
}

export function compilerConfigurationInvalidatedState(): CompilerConfigurationInvalidatedStatePatch {
  return {
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

export function compilerCosimInvalidatedState(): CompilerCosimInvalidatedStatePatch {
  return { cosimResult: null };
}

export function compilerIRLoadedState(
  response: IRBuildResponse,
): CompilerIRLoadedStatePatch {
  return {
    irErrors: response.errors,
    irText: response.ir_text,
    isSimulating: false,
  };
}

export function compilerSVLoadedState(response: SVEmitResponse): CompilerSVLoadedStatePatch {
  return { svSource: response.systemverilog };
}

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
