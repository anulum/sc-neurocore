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

/**
 * Run a custom ODE system the user wrote in the Studio.
 *
 * @param req - The equations, parameters, initial state and protocol.
 * @returns The trace, its spikes and the statistics the server computed.
 */
export const simulateODE = (req: Record<string, unknown>) => post<SimulateResponse>("/simulate", req);

/**
 * Run one catalogue neuron model.
 *
 * The server resolves the effective experiment — the model's own defaults fill
 * whatever the request leaves out — so the response, not the request, is what
 * an export has to state.
 *
 * @param req - The model name and whatever the caller wants to override.
 * @returns The trace, its spikes and the statistics the server computed.
 */
export const simulateModel = (req: Record<string, unknown>) => post<SimulateResponse>("/models/simulate", req);

/**
 * Run a balanced excitatory-inhibitory network.
 *
 * @param req - Population sizes, connectivity and drive.
 * @returns The raster, the population rates and the summary figures.
 */
export const simulateNetwork = (req: Record<string, unknown>) => post<NetworkResult>("/network/ei", req);

/**
 * Characterise a model: its f-I curve, its regimes and its response shape.
 *
 * @param req - The model and the sweep to characterise it over.
 * @returns The characterisation the server measured.
 */
export const fetchCharacterize = (req: Record<string, unknown>) => post<CharacterizeResponse>("/characterize", req);

/**
 * Run several configurations in one request, for the overlay view.
 *
 * One request rather than several keeps the runs on the same server-side
 * settings, which is what makes the traces comparable.
 *
 * @param configs - One configuration per trace, in the order to plot them.
 * @returns One result per configuration, in the same order.
 */
export const fetchMultiSimulate = (configs: Record<string, unknown>[]) => post<SimulateResponse[]>("/multi-simulate", configs);

/**
 * Hand a recorded voltage trace to the server for overlay and comparison.
 *
 * @param data - The samples and the timestep they were recorded at.
 * @returns The trace as the server holds it, with its derived statistics.
 */
export const importTrace = (data: { voltage: number[]; dt: number }) => post<ImportedTrace>("/import-trace", data);
