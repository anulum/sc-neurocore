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
  IRBuildResponse,
  IRVerifyResponse,
  SVEmitResponse,
  SVDirectResponse,
} from "./types";

export const compileVerilog = (req: Record<string, unknown>) => post<CompileResponse>("/compile", req);

export const compileModelVerilog = (req: ModelCompileRequest) => (
  post<CompileResponse>("/models/compile", req)
);

export const buildIR = (req: Record<string, unknown>) => post<IRBuildResponse>("/ir/build", req);

export const verifyIR = (irText: string) => post<IRVerifyResponse>("/ir/verify", { ir_text: irText });

export const emitSV = (irText: string) => post<SVEmitResponse>("/ir/emit-sv", { ir_text: irText });

export const emitSVDirect = (req: Record<string, unknown>) => post<SVDirectResponse>("/ir/emit-sv-direct", req);

export const fetchCosimDetail = (req: Record<string, unknown>) => post<PrecisionResponse>("/ir/cosim", req);
