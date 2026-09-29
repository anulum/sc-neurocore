// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Training configuration form

import { useEffect, useState } from "react";
import type { StudioProjectTrainingConfig } from "../studioProjectState";
import { readEventTrainingData } from "../studioEventTrainingData";
import TrainingEventInput from "./TrainingEventInput";
import TrainingPreregistrationInput from "./TrainingPreregistrationInput";
import { preregistrationProblem } from "../trainingPreregistration";
import { CONVERSION_DATASETS, conversionProblem } from "../trainingRequest";

/** Configuration controls sharing the Training Monitor's current store settings. */
export interface TrainingConfigurationProps {
  config: StudioProjectTrainingConfig;
  surrogates: { name: string }[];
  /** Hardware profiles a conversion run can be calibrated for. */
  targetProfiles?: { name: string; q_format: string }[];
  setConfig: <K extends keyof StudioProjectTrainingConfig>(
    key: K, value: StudioProjectTrainingConfig[K],
  ) => void;
  onReadyChange: (ready: boolean) => void;
}

/**
 * Check that a request can be submitted without replacing invalid form values.
 *
 * @param config - Current editable settings.
 * @returns Whether scalar settings, the declared criterion and event input are ready for server admission.
 */
function ready(config: StudioProjectTrainingConfig): boolean {
  if (![config.epochs, config.batch_size, config.timesteps].every(
    (value) => Number.isInteger(value) && value > 0,
  ) || !Number.isFinite(config.lr) || config.lr <= 0
    || !config.hidden.every((value) => Number.isInteger(value) && value > 0)
    || (config.seed !== undefined && (!Number.isInteger(config.seed) || config.seed < 0 || config.seed >= 2 ** 32))
    || (config.max_grad_norm !== undefined && (!Number.isFinite(config.max_grad_norm) || config.max_grad_norm < 0))
    || (config.preregistration !== undefined
      && preregistrationProblem(config.preregistration, config.model_kind) !== null)
    || conversionProblem(config) !== null) {
    return false;
  }
  try {
    readEventTrainingData(config.event_data, config.dataset, config.timesteps);
    return true;
  } catch {
    return false;
  }
}

/**
 * Edit static or temporal training settings, retaining declared replay values.
 *
 * @param props - Current settings, setter, surrogate choices and readiness callback.
 * @returns The configuration controls and event declaration editor.
 */
export default function TrainingConfiguration({
  config, surrogates, targetProfiles = [], setConfig, onReadyChange,
}: TrainingConfigurationProps) {
  const [dirtyInput, setDirtyInput] = useState(false);
  const valid = ready(config) && !dirtyInput;
  const conversion = config.model_kind === "qcfs_conversion";
  const problem = conversionProblem(config);
  useEffect(() => { onReadyChange(valid); }, [valid, onReadyChange]);
  /**
   * Switch the run's kind, moving a conversion run off an event dataset it cannot encode.
   *
   * @param kind - The chosen kind.
   */
  const chooseKind = (kind: string) => {
    if (kind !== "qcfs_conversion") {
      setConfig("model_kind", undefined);
      return;
    }
    setConfig("model_kind", "qcfs_conversion");
    if (!CONVERSION_DATASETS.includes(config.dataset)) {
      setDirtyInput(false);
      setConfig("event_data", undefined);
      setConfig("dataset", "synthetic");
    }
  };
  return <>
        <div style={{
          padding: "8px 12px", borderBottom: "1px solid var(--border)",
          display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 6,
          fontSize: 10,
        }}>
          <label style={{ color: "var(--text-secondary)" }}>
            Model
            <select aria-label="Model" value={config.model_kind ?? "spiking"} onChange={(e) => { chooseKind(e.target.value); }}
              style={{ display: "block", width: "100%", fontSize: 10 }}>
              <option value="spiking">Spiking network (surrogate gradients)</option>
              <option value="qcfs_conversion">QCFS ANN, converted to IF</option>
            </select>
          </label>
          <label style={{ color: "var(--text-secondary)" }}>
            Dataset
            <select aria-label="Dataset" value={config.dataset} onChange={(e) => { setDirtyInput(false); setConfig("event_data", undefined); setConfig("dataset", e.target.value); }}
              style={{ display: "block", width: "100%", fontSize: 10 }}>
              <option value="synthetic">Synthetic (64D, fast)</option>
              <option value="mnist">MNIST (784D)</option>
              {!conversion && <>
                <option value="nmnist">N-MNIST events</option>
                <option value="shd">SHD events</option>
                <option value="dvs_cifar10">DVS-CIFAR10 events</option>
              </>}
            </select>
          </label>
          <label style={{ color: "var(--text-secondary)" }}>
            Epochs
            <input aria-label="Epochs" type="number" value={config.epochs} min={1} max={100}
              onChange={(e) => { setConfig("epochs", Number(e.target.value)); }}
              style={{ display: "block", width: "100%", fontSize: 10 }} />
          </label>
          <label style={{ color: "var(--text-secondary)" }}>
            Batch Size
            <input aria-label="Batch Size" type="number" value={config.batch_size} min={1} step={1}
              onChange={(e) => { setConfig("batch_size", Number(e.target.value)); }}
              style={{ display: "block", width: "100%", fontSize: 10 }} />
          </label>
          <label style={{ color: "var(--text-secondary)" }}>
            Learning Rate
            <input aria-label="Learning Rate" type="number" value={config.lr} min={0.0001} max={0.1} step={0.0001}
              onChange={(e) => { setConfig("lr", Number(e.target.value)); }}
              style={{ display: "block", width: "100%", fontSize: 10 }} />
          </label>
          <label style={{ color: "var(--text-secondary)" }}>
            Timesteps
            <input aria-label="Timesteps" type="number" value={config.timesteps} min={1} step={1}
              onChange={(e) => { setConfig("timesteps", Number(e.target.value)); }}
              style={{ display: "block", width: "100%", fontSize: 10 }} />
          </label>
          {conversion && <label style={{ color: "var(--text-secondary)" }}>
            Target
            <select aria-label="Target" value={config.target_profile ?? ""}
              onChange={(e) => { setConfig("target_profile", e.target.value === "" ? undefined : e.target.value); }}
              style={{ display: "block", width: "100%", fontSize: 10 }}>
              <option value="">No target calibration</option>
              {config.target_profile !== undefined
                && !targetProfiles.some((profile) => profile.name === config.target_profile)
                && <option value={config.target_profile}>{config.target_profile}</option>}
              {targetProfiles.map((profile) => (
                <option key={profile.name} value={profile.name}>{`${profile.name} (${profile.q_format})`}</option>
              ))}
            </select>
          </label>}
          {!conversion && <>
          <label style={{ color: "var(--text-secondary)" }}>
            Surrogate
            <select aria-label="Surrogate" value={config.surrogate} onChange={(e) => { setConfig("surrogate", e.target.value); }}
              style={{ display: "block", width: "100%", fontSize: 10 }}>
              {(surrogates.length > 0 ? surrogates : [
                { name: "atan_surrogate" }, { name: "fast_sigmoid" }, { name: "superspike" },
                { name: "sigmoid_surrogate" }, { name: "straight_through" }, { name: "triangular" },
              ]).map((s) => (
                <option key={s.name} value={s.name}>{s.name.replace(/_/g, " ")}</option>
              ))}
            </select>
          </label>
          <label style={{ color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 4 }}>
            <input type="checkbox" checked={config.learn_beta}
              onChange={(e) => { setConfig("learn_beta", e.target.checked); }} />
            Learn beta
          </label>
          <label style={{ color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 4 }}>
            <input type="checkbox" checked={config.learn_threshold}
              onChange={(e) => { setConfig("learn_threshold", e.target.checked); }} />
            Learn threshold
          </label>
          </>}
          <label style={{ color: "var(--text-secondary)" }}>
            Seed
            <input aria-label="Seed" type="number" value={config.seed ?? ""} min={0} max={2 ** 32 - 1} step={1}
              placeholder="Server default"
              onChange={(e) => { setConfig("seed", e.target.value === "" ? undefined : Number(e.target.value)); }} />
          </label>
          <label style={{ color: "var(--text-secondary)" }}>
            Gradient norm limit
            <input aria-label="Gradient norm limit" type="number" value={config.max_grad_norm ?? ""} min={0} step="any"
              placeholder="Server default"
              onChange={(e) => { setConfig("max_grad_norm", e.target.value === "" ? undefined : Number(e.target.value)); }} />
          </label>
        </div>
    {conversion && <p style={{ padding: "0 12px", fontSize: 10 }}>
      Trains an ANN with QCFS activations using Timesteps as their step budget, converts it to an
      integrate-and-fire network with the same budget and reports the converted network&apos;s
      validation accuracy beside the source&apos;s.
    </p>}
    {problem !== null && <p role="alert" style={{ padding: "0 12px" }}>{problem}</p>}
    <TrainingPreregistrationInput value={config.preregistration} modelKind={config.model_kind}
      onChange={(value) => { setConfig("preregistration", value); }} />
    {config.dataset !== "synthetic" && config.dataset !== "mnist" && <TrainingEventInput
      key={config.dataset} config={config} setConfig={setConfig} onDirtyChange={setDirtyInput}
    />}
    {!valid && <p role="status" style={{ padding: "0 12px" }}>
      Correct the training settings and apply a matching event input before starting.
    </p>}
  </>;
}
