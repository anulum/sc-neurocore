// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio simulation request builders

export type StudioSimulationSourceMode = "model" | "ode";
export type StudioSimulationRequest = Record<string, unknown>;
export type StudioSimulationTrial = "replay" | "fresh";

export interface StudioSimulationConfigInput {
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
  /** Sine-protocol frequency; always sent so the effective config is explicit. */
  frequencyHz: number;
  /** Explicit seed of a stochastic run; ``null`` leaves the model or playground default. */
  seed: number | null;
  /** ``replay`` (cacheable, deterministic) or ``fresh`` (independent stochastic trial). */
  trial: StudioSimulationTrial;
}

/**
 * The explicit randomness and protocol fields every simulation request carries.
 */
export function studioExperimentFields(input: StudioSimulationConfigInput): StudioSimulationRequest {
  const fields: StudioSimulationRequest = {
    frequency_hz: input.frequencyHz,
    trial: input.trial,
  };
  if (input.seed !== null) {
    fields.seed = input.seed;
  }
  return fields;
}

export interface StudioCodegenRequestInput extends StudioSimulationConfigInput {
  sourceMode: StudioSimulationSourceMode;
}

export interface StudioBifurcationSweepInput {
  sweepParam: string;
  parameterValue: number;
}

export interface StudioHeatmapSweepInput {
  sweepParamX: string;
  parameterValueX: number;
  sweepParamY: string;
  parameterValueY: number;
}

export function studioSimulationConfig(input: StudioSimulationConfigInput): StudioSimulationRequest {
  if (input.sourceMode === "model" && input.selectedModelName) {
    return {
      model_name: input.selectedModelName,
      params: input.modelParams,
      dt: input.dt,
      duration: input.duration,
      current: input.current,
      protocol: input.protocol,
      ...studioExperimentFields(input),
    };
  }
  return {
    equations: input.equations,
    threshold: input.threshold || null,
    reset: input.reset || null,
    params: input.odeParams,
    init: input.odeInit,
    dt: input.dt,
    duration: input.duration,
    current: input.current,
    protocol: input.protocol,
    ...studioExperimentFields(input),
  };
}

export function studioFICurveRequest(
  config: StudioSimulationRequest,
  current: number,
): StudioSimulationRequest {
  return {
    ...config,
    i_min: 0,
    i_max: Math.abs(current) * 2 || 50,
    i_steps: 25,
  };
}

export function studioBifurcationRequest(
  config: StudioSimulationRequest,
  sweep: StudioBifurcationSweepInput,
): StudioSimulationRequest {
  return {
    ...config,
    sweep_param: sweep.sweepParam,
    sweep_min: sweep.parameterValue * 0.2,
    sweep_max: sweep.parameterValue * 3,
    sweep_steps: 40,
  };
}

export function studioHeatmapRequest(
  config: StudioSimulationRequest,
  sweep: StudioHeatmapSweepInput,
): StudioSimulationRequest {
  return {
    ...config,
    param_x: sweep.sweepParamX,
    x_min: sweep.parameterValueX * 0.2,
    x_max: sweep.parameterValueX * 3,
    x_steps: 15,
    param_y: sweep.sweepParamY,
    y_min: sweep.parameterValueY * 0.2,
    y_max: sweep.parameterValueY * 3,
    y_steps: 15,
  };
}

export const DEFAULT_PRECISION_Q_FORMAT = "Q8.8";

/**
 * Float64 versus bit-true fixed-point comparison of the custom ODE. The
 * protocol, sine frequency and word format are explicit so the three runs the
 * server compares are the configured experiment, not a constant-current default.
 */
export function studioPrecisionRequest(
  input: StudioSimulationConfigInput,
  qFormat: string = DEFAULT_PRECISION_Q_FORMAT,
): StudioSimulationRequest {
  return {
    equations: input.equations,
    threshold: input.threshold,
    reset: input.reset,
    params: input.odeParams,
    init: input.odeInit,
    dt: input.dt,
    duration: input.duration,
    current: input.current,
    protocol: input.protocol,
    frequency_hz: input.frequencyHz,
    q_format: qFormat,
  };
}

export interface StudioNullclineRequestInput {
  equations: string[];
  odeParams: Record<string, number>;
  odeInit: Record<string, number>;
  protocol: string;
  current: number;
  ranges: Record<string, [number, number]>;
  gridSize: number;
}

/**
 * Nullclines of the first two ODE variables; every further variable is held at
 * its initial value and the drift field is evaluated at the configured current
 * when the protocol is constant (otherwise at zero input).
 */
export function studioNullclineRequest(input: StudioNullclineRequestInput): StudioSimulationRequest {
  const vars = Object.keys(input.odeInit);
  const held: Record<string, number> = {};
  for (const name of vars.slice(2)) {
    held[name] = input.odeInit[name];
  }
  return {
    equations: input.equations,
    params: input.odeParams,
    var_names: vars.slice(0, 2),
    ranges: input.ranges,
    grid_size: input.gridSize,
    current: input.protocol === "constant" ? input.current : 0,
    held,
  };
}

export function studioCodegenRequest(input: StudioCodegenRequestInput): StudioSimulationRequest {
  return {
    mode: input.sourceMode,
    model_name: input.sourceMode === "model" ? input.selectedModelName : null,
    equations: input.sourceMode === "ode" ? input.equations : null,
    threshold: input.threshold,
    reset: input.reset,
    params: input.sourceMode === "model" ? input.modelParams : input.odeParams,
    init: input.sourceMode === "ode" ? input.odeInit : null,
    dt: input.dt,
    duration: input.duration,
    current: input.current,
  };
}

export function studioFrequencyResponseRequest(
  config: StudioSimulationRequest,
  current: number,
): StudioSimulationRequest {
  return {
    ...config,
    amplitude: Math.abs(current) || 10,
    freq_min: 1,
    freq_max: 200,
    n_freqs: 20,
  };
}
