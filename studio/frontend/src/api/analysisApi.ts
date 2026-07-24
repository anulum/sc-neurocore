// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: analysis endpoints.
import { post } from "./http";
import type {
  PrecisionResponse,
  NullclineResponse,
  CompareResponse,
  FreqResponse,
  AnalysisJobRequestBody,
  AnalysisJobReceipt,
} from "./types";

export const submitAnalysisJob = (request: AnalysisJobRequestBody) =>
  post<AnalysisJobReceipt>("/analysis/jobs", request);

export const fetchNullclines = (req: Record<string, unknown>) => post<NullclineResponse>("/nullclines", req);

export const fetchPrecision = (req: Record<string, unknown>) => post<PrecisionResponse>("/precision", req);

export const fetchCompare = (a: Record<string, unknown>, b: Record<string, unknown>) => post<CompareResponse>("/compare", { config_a: a, config_b: b });

export const fetchFreqResponse = (req: Record<string, unknown>) => post<FreqResponse>("/freq-response", req);

export const fetchCodegen = (req: Record<string, unknown>) => post<{ script: string; oneliner: string }>("/codegen", req);
