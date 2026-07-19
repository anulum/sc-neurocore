// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Pure Studio simulation config input extraction

/**
 * Field-for-field assembly of {@link StudioSimulationConfigInput} from a narrow
 * structural source. No store imports, defaults, normalisation, or API calls.
 */

import type {
  StudioSimulationConfigInput,
  StudioSimulationSourceMode,
} from "./studioSimulationConfig";

/**
 * Narrow structural source for simulation config extraction.
 * Contains only fields required to produce StudioSimulationConfigInput.
 */
export interface StudioSimulationConfigSource {
  sourceMode: StudioSimulationSourceMode;
  selectedModelName: string;
  modelParams: Record<string, number>;
  equations: string[];
  threshold: string;
  reset: string;
  odeParams: Record<string, number>;
  odeInit: Record<string, number>;
  dt: number;
  duration: number;
  current: number;
  protocol: string;
}

/**
 * Build StudioSimulationConfigInput with exact field-for-field copy semantics.
 * Map and array fields are returned by reference without mutation.
 */
export function studioSimulationConfigInput(
  source: StudioSimulationConfigSource,
): StudioSimulationConfigInput {
  return {
    sourceMode: source.sourceMode,
    selectedModelName: source.selectedModelName,
    modelParams: source.modelParams,
    equations: source.equations,
    threshold: source.threshold,
    reset: source.reset,
    odeParams: source.odeParams,
    odeInit: source.odeInit,
    dt: source.dt,
    duration: source.duration,
    current: source.current,
    protocol: source.protocol,
  };
}
