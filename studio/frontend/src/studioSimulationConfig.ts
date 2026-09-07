// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio simulation request builders

/**
 * Turning what the panel holds into the body of a request.
 *
 * Two things are deliberate throughout. **The experiment fields are always
 * sent** -- the protocol, the sine frequency, and whether the trial replays or
 * draws fresh -- so a request states the experiment it is rather than relying
 * on a server default that may differ between deployments. A seed is the
 * exception: it is sent only when the reader set one, because sending `null`
 * and sending nothing mean different things to the server, and only the second
 * means "choose for me".
 *
 * **The sweep ranges are derived from the value the reader is already using**,
 * from a fifth of it to three times it. That is a starting window around a
 * working point, not a claim about where the interesting behaviour is, and
 * every bound is visible and editable once the sweep has run.
 */

/** Whether a run is driven by a catalogue model or by an ODE. */
export type StudioSimulationSourceMode = "model" | "ode";
/**
 * A request body, as an open record. The routes differ enough in shape that
 * typing each one here would duplicate the server's own contract; what this
 * module guarantees is which fields are set, not that the whole body is valid.
 */
export type StudioSimulationRequest = Record<string, unknown>;
/**
 * Whether a run replays deterministically or draws a fresh stochastic trial.
 * `replay` is cacheable; `fresh` deliberately is not.
 */
export type StudioSimulationTrial = "replay" | "fresh";

/** Everything a run is built from, as the panel holds it. */
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
 * The randomness and protocol fields every request carries.
 *
 * @param input - The configuration the panel holds.
 * @returns The fields. The seed is present only when the reader set one:
 *   omitting it asks the server to choose, and sending `null` does not.
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

/**
 * What a codegen or export request is built from. Identical to a run's input:
 * generated code must describe the run it was generated from.
 */
export interface StudioCodegenRequestInput extends StudioSimulationConfigInput {
  sourceMode: StudioSimulationSourceMode;
}

/** A one-parameter sweep: which parameter, and its current value. */
export interface StudioBifurcationSweepInput {
  sweepParam: string;
  parameterValue: number;
}

/** A two-parameter sweep: both parameters, and their current values. */
export interface StudioHeatmapSweepInput {
  sweepParamX: string;
  parameterValueX: number;
  sweepParamY: string;
  parameterValueY: number;
}

/**
 * Build the body of a simulation request.
 *
 * A model run and an ODE run send different fields, not one shape with
 * nulls in it: the routes are fail-closed, and a null of the other branch
 * is a field the server rejects rather than ignores.
 *
 * @param input - The configuration the panel holds.
 * @returns The request body for whichever branch is in force.
 */
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

/**
 * Extend a run into an f-I curve sweep.
 *
 * The current range runs from zero to twice the configured current, or to
 * 50 when that is zero -- a curve of a single point is not a curve.
 *
 * @param config - The run's request body.
 * @param current - The current the run is configured at.
 * @returns The sweep's request body.
 */
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

/**
 * Extend a run into a one-parameter bifurcation sweep.
 *
 * @param config - The run's request body.
 * @param sweep - The parameter and its current value.
 * @returns The sweep's request body, spanning a fifth to three times that
 *   value.
 */
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

/**
 * Extend a run into a two-parameter heatmap.
 *
 * @param config - The run's request body.
 * @param sweep - Both parameters and their current values.
 * @returns The sweep's request body. Fifteen steps per axis, because the
 *   cost is their product and 225 runs is already a wait.
 */
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

/** The fixed-point word format a precision comparison uses by default. */
export const DEFAULT_PRECISION_Q_FORMAT = "Q8.8";

/**
 * Build the request that compares float64 against bit-true fixed point.
 *
 * The protocol, the sine frequency and the word format are all stated, so the
 * runs the server compares are the experiment the reader configured rather
 * than a constant-current default -- a precision comparison against a
 * different stimulus answers a question nobody asked.
 *
 * @param input - The configuration the panel holds.
 * @param qFormat - The fixed-point word format to compare against.
 * @returns The request body.
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

/** What a nullcline computation needs: the system, its ranges, its grid. */
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
 * Build the request that computes nullclines.
 *
 * The first two variables span the plane; every further one is held at its
 * initial value, because a nullcline plot is two-dimensional and the rest have
 * to be somewhere. The drift field is evaluated at the configured current only
 * for a constant protocol -- for a time-varying one there is no single current
 * to evaluate at, and zero is the honest choice.
 *
 * @param input - The system, its ranges and the grid to evaluate on.
 * @returns The request body.
 */
export function studioNullclineRequest(input: StudioNullclineRequestInput): StudioSimulationRequest {
  const vars = Object.keys(input.odeInit);
  const held: Record<string, number> = {};
  for (const name of vars.slice(2)) {
    const value = input.odeInit[name];
    // `vars` is this object's own key list, so the lookup cannot miss; the
    // guard is here because the index type cannot say that, and skipping is
    // the only sane answer if the object is ever mutated between the two.
    if (value !== undefined) held[name] = value;
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

/**
 * Build the body of an export request.
 *
 * An export describes the same experiment as the run it comes from, so it
 * reuses `studioSimulationConfig` rather than restating the fields. Only the
 * branch's own fields are sent: the export routes are fail-closed, and a null
 * belonging to the other branch is a rejected field, not an empty one.
 *
 * @param input - The configuration the panel holds.
 * @returns The request body, with the mode it was built for.
 */
export function studioExperimentExportRequest(
  input: StudioCodegenRequestInput,
): StudioSimulationRequest {
  return { mode: input.sourceMode, ...studioSimulationConfig(input) };
}

/**
 * Extend a run into a frequency-response sweep.
 *
 * @param config - The run's request body.
 * @param current - The current the run is configured at, which becomes the
 *   drive amplitude; a zero current would drive nothing, so it falls back.
 * @returns The sweep's request body.
 */
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
