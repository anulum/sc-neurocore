// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio simulation input state helpers

/**
 * The state patches that change what an experiment *is*, before it is run.
 *
 * Every setter returns only the field it sets, so changing the timestep cannot
 * silently reset the protocol. The two that take a `current` argument —
 * `numberRecordEntryState` and `networkParamState` — copy the record they are
 * given rather than reading the store, which is what keeps them pure and lets
 * a caller apply two changes to one snapshot without the second losing the
 * first.
 */
import type { ModelDetail } from "./api/client";
import { modelDefaultParameters } from "./modelSelectionStoreState";

/** The tabs an input change may move the Studio to. */
export type StudioInputViewTab =
  "trace" | "phase" | "isi" | "fi-curve" | "bifurcation" | "sensitivity" |
  "precision" | "heatmap" | "verilog" | "code" | "compare" | "freq" |
  "sta" | "characterize" | "multi" | "network" | "ir" | "synth" |
  "train" | "canvas" | "delays" | "admin" | "candidate" | "fit" | "review";

/** The experiment now comes from a catalogue model, or from typed equations. */
export interface StudioSourceModeStatePatch {
  sourceMode: "model" | "ode";
}

/** The typed equations changed. */
export interface StudioEquationsStatePatch {
  equations: string[];
}

/** The spike threshold expression changed. */
export interface StudioThresholdStatePatch {
  threshold: string;
}

/** The post-spike reset expression changed. */
export interface StudioResetStatePatch {
  reset: string;
}

/**
 * One entry of one numeric record changed, the rest of it carried over.
 *
 * Keyed loosely because the field is chosen at the call site — `modelParams`,
 * `odeInit` or `odeParams` — and the builder's own parameter is what narrows
 * it to those three.
 */
export type StudioNumberRecordPatch = Record<string, Record<string, number>>;

/** The integration timestep changed. */
export interface StudioDtStatePatch {
  dt: number;
}

/** The run duration changed. */
export interface StudioDurationStatePatch {
  duration: number;
}

/** The injected current changed. */
export interface StudioCurrentStatePatch {
  current: number;
}

/** The drive protocol changed. */
export interface StudioProtocolStatePatch {
  protocol: string;
}

/** The visible tab changed. */
export interface StudioActiveTabStatePatch {
  activeTab: StudioInputViewTab;
}

/** The catalogue search text changed. */
export interface StudioModelFilterStatePatch {
  modelFilter: string;
}

/** The parameter a sweep varies changed. */
export interface StudioSweepParamStatePatch {
  sweepParam: string;
}

/** The second parameter a two-dimensional sweep varies changed. */
export interface StudioSweepParamYStatePatch {
  sweepParamY: string;
}

/** Everything a balanced excitatory-inhibitory network run is defined by. */
export interface StudioNetworkParams {
  ext_rate: number;
  n_exc: number;
  n_inh: number;
  p_conn: number;
  w_ee: number;
  w_ei: number;
  w_ie: number;
  w_ii: number;
}

/** One network parameter changed, the rest carried over. */
export interface StudioNetworkParamsStatePatch {
  networkParams: StudioNetworkParams;
}

/** A newly selected model's own defaults, replacing the previous experiment. */
export interface StudioModelDefaultsStatePatch {
  current: 10;
  dt: number;
  duration: 100;
  modelParams: Record<string, number>;
}

/**
 * Switch between a catalogue model and typed equations.
 *
 * @param sourceMode - Which the experiment now comes from.
 * @returns The patch.
 */
export function sourceModeState(sourceMode: "model" | "ode"): StudioSourceModeStatePatch {
  return { sourceMode };
}

/**
 * Set the typed equations.
 *
 * @param equations - The equations, one per line.
 * @returns The patch.
 */
export function equationsState(equations: string[]): StudioEquationsStatePatch {
  return { equations: [...equations] };
}

/**
 * Set the spike threshold expression.
 *
 * @param threshold - The expression.
 * @returns The patch.
 */
export function thresholdState(threshold: string): StudioThresholdStatePatch {
  return { threshold };
}

/**
 * Set the post-spike reset expression.
 *
 * @param reset - The expression.
 * @returns The patch.
 */
export function resetState(reset: string): StudioResetStatePatch {
  return { reset };
}

/**
 * Change one entry of one numeric record, keeping the rest.
 *
 * The record is passed in rather than read from the store, so two changes can
 * be applied to one snapshot without the second losing the first.
 *
 * @param field - Which record to change.
 * @param current - The record as it stands.
 * @param key - The entry to change.
 * @param value - Its new value.
 * @returns The patch.
 */
export function numberRecordEntryState(
  field: "modelParams" | "odeInit" | "odeParams",
  current: Record<string, number>,
  key: string,
  value: number,
): StudioNumberRecordPatch {
  return { [field]: { ...current, [key]: value } };
}

/**
 * Set the integration timestep.
 *
 * @param dt - The timestep in milliseconds.
 * @returns The patch.
 */
export function dtState(dt: number): StudioDtStatePatch {
  return { dt };
}

/**
 * Set the run duration.
 *
 * @param duration - The duration in milliseconds.
 * @returns The patch.
 */
export function durationState(duration: number): StudioDurationStatePatch {
  return { duration };
}

/**
 * Set the injected current.
 *
 * @param current - The current in nanoamps.
 * @returns The patch.
 */
export function currentState(current: number): StudioCurrentStatePatch {
  return { current };
}

/**
 * Set the drive protocol.
 *
 * @param protocol - The protocol's name.
 * @returns The patch.
 */
export function protocolState(protocol: string): StudioProtocolStatePatch {
  return { protocol };
}

/** The drive frequency changed. */
export interface StudioFrequencyHzStatePatch {
  frequencyHz: number;
}

/** The random seed changed, or was cleared to let the server draw one. */
export interface StudioSeedStatePatch {
  seed: number | null;
}

/** Whether a stochastic run replays its seed or draws a new one. */
export interface StudioTrialStatePatch {
  trial: "replay" | "fresh";
}

/**
 * Set the drive frequency.
 *
 * @param frequencyHz - The frequency in hertz.
 * @returns The patch.
 */
export function frequencyHzState(frequencyHz: number): StudioFrequencyHzStatePatch {
  return { frequencyHz };
}

/**
 * Set or clear the random seed.
 *
 * @param seed - The seed, or `null` to let the server draw one.
 * @returns The patch.
 */
export function seedState(seed: number | null): StudioSeedStatePatch {
  return { seed };
}

/**
 * Choose whether a stochastic run replays its seed or draws a new one.
 *
 * @param trial - Which.
 * @returns The patch.
 */
export function trialState(trial: "replay" | "fresh"): StudioTrialStatePatch {
  return { trial };
}

/**
 * Move to a tab.
 *
 * @param activeTab - The tab.
 * @returns The patch.
 */
export function activeTabState(activeTab: StudioInputViewTab): StudioActiveTabStatePatch {
  return { activeTab };
}

/**
 * Set the catalogue search text.
 *
 * @param modelFilter - The text.
 * @returns The patch.
 */
export function modelFilterState(modelFilter: string): StudioModelFilterStatePatch {
  return { modelFilter };
}

/**
 * Choose the parameter a sweep varies.
 *
 * @param sweepParam - The parameter's name.
 * @returns The patch.
 */
export function sweepParamState(sweepParam: string): StudioSweepParamStatePatch {
  return { sweepParam };
}

/**
 * Choose the second parameter a two-dimensional sweep varies.
 *
 * @param sweepParamY - The parameter's name.
 * @returns The patch.
 */
export function sweepParamYState(sweepParamY: string): StudioSweepParamYStatePatch {
  return { sweepParamY };
}

/**
 * Change one network parameter, keeping the rest.
 *
 * @param current - The parameters as they stand.
 * @param key - The parameter to change.
 * @param value - Its new value.
 * @returns The patch.
 */
export function networkParamState<K extends keyof StudioNetworkParams>(
  current: StudioNetworkParams,
  key: K,
  value: StudioNetworkParams[K],
): StudioNetworkParamsStatePatch {
  return {
    networkParams: {
      ...current,
      [key]: value,
    },
  };
}

/**
 * Replace the experiment with a newly selected model's own defaults.
 *
 * The model contributes its timestep and its parameters; the current and the
 * duration are the Studio's own starting points, because a model's contract
 * declares neither.
 *
 * @param modelDetail - The model.
 * @returns The patch.
 */
export function modelDefaultsState(modelDetail: ModelDetail): StudioModelDefaultsStatePatch {
  return {
    current: 10,
    dt: modelDetail.dt,
    duration: 100,
    modelParams: modelDefaultParameters(modelDetail),
  };
}
