// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: simulation endpoints.
import { post } from "./http";
import type {
  SimulateResponse,
  NetworkResult,
  CharacterizeResponse,
  ImportedTrace,
} from "./types";

export const simulateODE = (req: Record<string, unknown>) => post<SimulateResponse>("/simulate", req);

export const simulateModel = (req: Record<string, unknown>) => post<SimulateResponse>("/models/simulate", req);

export const simulateNetwork = (req: Record<string, unknown>) => post<NetworkResult>("/network/ei", req);

export const fetchCharacterize = (req: Record<string, unknown>) => post<CharacterizeResponse>("/characterize", req);

export const fetchMultiSimulate = (configs: Record<string, unknown>[]) => post<SimulateResponse[]>("/multi-simulate", configs);

export const importTrace = (data: { voltage: number[]; dt: number }) => post<ImportedTrace>("/import-trace", data);

