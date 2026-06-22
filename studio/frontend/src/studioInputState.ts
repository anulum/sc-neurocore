// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio simulation input state helpers
import type { ModelDetail } from "./api/client";
import { modelDefaultParameters } from "./modelSelectionStoreState";

export type StudioInputViewTab =
  "trace" | "phase" | "isi" | "fi-curve" | "bifurcation" | "sensitivity" |
  "precision" | "heatmap" | "verilog" | "code" | "compare" | "freq" |
  "sta" | "characterize" | "multi" | "network" | "ir" | "synth" |
  "train" | "canvas" | "delays" | "admin";

export interface StudioSourceModeStatePatch {
  sourceMode: "model" | "ode";
}

export interface StudioEquationsStatePatch {
  equations: string[];
}

export interface StudioThresholdStatePatch {
  threshold: string;
}

export interface StudioResetStatePatch {
  reset: string;
}

export interface StudioNumberRecordPatch {
  [key: string]: Record<string, number>;
}

export interface StudioDtStatePatch {
  dt: number;
}

export interface StudioDurationStatePatch {
  duration: number;
}

export interface StudioCurrentStatePatch {
  current: number;
}

export interface StudioProtocolStatePatch {
  protocol: string;
}

export interface StudioActiveTabStatePatch {
  activeTab: StudioInputViewTab;
}

export interface StudioModelFilterStatePatch {
  modelFilter: string;
}

export interface StudioSweepParamStatePatch {
  sweepParam: string;
}

export interface StudioSweepParamYStatePatch {
  sweepParamY: string;
}

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

export interface StudioNetworkParamsStatePatch {
  networkParams: StudioNetworkParams;
}

export interface StudioModelDefaultsStatePatch {
  current: 10;
  dt: number;
  duration: 100;
  modelParams: Record<string, number>;
}

export function sourceModeState(sourceMode: "model" | "ode"): StudioSourceModeStatePatch {
  return { sourceMode };
}

export function equationsState(equations: string[]): StudioEquationsStatePatch {
  return { equations: [...equations] };
}

export function thresholdState(threshold: string): StudioThresholdStatePatch {
  return { threshold };
}

export function resetState(reset: string): StudioResetStatePatch {
  return { reset };
}

export function numberRecordEntryState(
  field: "modelParams" | "odeInit" | "odeParams",
  current: Record<string, number>,
  key: string,
  value: number,
): StudioNumberRecordPatch {
  return { [field]: { ...current, [key]: value } };
}

export function dtState(dt: number): StudioDtStatePatch {
  return { dt };
}

export function durationState(duration: number): StudioDurationStatePatch {
  return { duration };
}

export function currentState(current: number): StudioCurrentStatePatch {
  return { current };
}

export function protocolState(protocol: string): StudioProtocolStatePatch {
  return { protocol };
}

export function activeTabState(activeTab: StudioInputViewTab): StudioActiveTabStatePatch {
  return { activeTab };
}

export function modelFilterState(modelFilter: string): StudioModelFilterStatePatch {
  return { modelFilter };
}

export function sweepParamState(sweepParam: string): StudioSweepParamStatePatch {
  return { sweepParam };
}

export function sweepParamYState(sweepParamY: string): StudioSweepParamYStatePatch {
  return { sweepParamY };
}

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

export function modelDefaultsState(modelDetail: ModelDetail): StudioModelDefaultsStatePatch {
  return {
    current: 10,
    dt: modelDetail.dt,
    duration: 100,
    modelParams: modelDefaultParameters(modelDetail),
  };
}
