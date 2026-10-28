// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Authored equation compiler controls

import type { CSSProperties } from "react";
import {
  defaultEquationIrConfiguration, equationIrChoice, EQUATION_METHODS, EQUATION_DETECTIONS,
  EQUATION_FUNCTION_PROFILES, EQUATION_OVERFLOWS, EQUATION_RANDOM_PROFILES, EQUATION_ROUNDINGS,
  EQUATION_THRESHOLD_GENERATORS, EQUATION_UNIT_MODES,
  type EquationIrModelDraft, type EquationIrProfileDraft,
} from "../equationIrConfig";
import { useStudioStore } from "../stores/studio";

const controlStyle: CSSProperties = {
  width: "100%", boxSizing: "border-box", background: "var(--bg-primary)",
  color: "var(--text-primary)", border: "1px solid var(--control-border)",
  borderRadius: "var(--radius)", padding: "4px 6px", fontFamily: "var(--font-mono)",
};
const gridStyle: CSSProperties = {
  display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: 10,
};
const profileFields = [
  { key: "module_name", label: "Module name", help: "Native HDL module identifier." },
  { key: "data_width", label: "Total scalar bits", help: "Includes the fractional bits." },
  { key: "fraction", label: "Fractional bits", help: "Fixed-point fractional precision." },
  { key: "pipeline_stages", label: "Product pipeline stages", help: "0 disables global product registers." },
  { key: "pipeline_points", label: "Product pipeline points", help: "Optional comma-separated product names, instead of global stages." },
] as const;
const modelFields = [
  { key: "substeps", label: "Substeps", help: "Multiple substeps require crossing detection without reset rules." },
  { key: "rate_expression", label: "Escape-rate expression", help: "Used with escape_rate detection." },
  { key: "probability_expression", label: "Spike probability expression", help: "Used with poisson detection." },
  { key: "spike_value", label: "Graded spike expression", help: "Blank emits Boolean spikes only." },
  { key: "rng_seed", label: "Compiler random key", help: "Exact unsigned 64-bit decimal text; separate from the simulation seed." },
  { key: "random_stream", label: "Compiler random stream", help: "Unsigned integer from 0 through 4294967295." },
  { key: "constant_names", label: "Immutable constant names", help: "Comma-separated playground parameters to freeze at their current values." },
] as const;
const unitFields = [
  { key: "state_units", label: "State unit declarations", help: '[{"name":"v","unit":"millivolt"}]', multiline: true },
  { key: "parameter_units", label: "Parameter and constant unit declarations", help: '[{"name":"tau","unit":"millisecond"}]', multiline: true },
  { key: "dt_unit", label: "Timestep unit", help: "For example millisecond; magnitude comes from the playground timestep.", multiline: false },
  { key: "input_unit", label: "Drive unit", help: "Physical unit of the equation's I input.", multiline: false },
  { key: "input_magnitude", label: "Drive unit declaration magnitude", help: "Magnitude of the quantity declaring the I input's unit.", multiline: false },
] as const;

/** One losslessly editable setting with its scientific meaning visible. */
interface TextSettingProps {
  label: string;
  value: string;
  help: string;
  multiline?: boolean;
  onChange: (value: string) => void;
}
/**
 * Render text without prematurely coercing an incomplete numerical draft.
 *
 * @returns Labelled control and field-specific help.
 */
function TextSetting({ label, value, help, multiline = false, onChange }: TextSettingProps) {
  return <label style={{ display: "flex", flexDirection: "column", gap: 3 }}>
    <span>{label}</span>
    {multiline
      ? <textarea aria-label={label} rows={2} style={controlStyle} value={value}
        onChange={(event) => { onChange(event.target.value); }} />
      : <input aria-label={label} style={controlStyle} value={value}
        onChange={(event) => { onChange(event.target.value); }} />}
    <small style={{ color: "var(--text-muted)" }}>{help}</small>
  </label>;
}

/** A selector whose values are checked against maintained literal options. */
interface ChoiceSettingProps<T extends string> {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (value: T) => void;
}
/**
 * Render a typed compiler choice without casting a DOM string.
 *
 * @returns Labelled selector.
 */
function ChoiceSetting<T extends string>({ label, value, options, onChange }: ChoiceSettingProps<T>) {
  return <label style={{ display: "flex", flexDirection: "column", gap: 3 }}>
    <span>{label}</span>
    <select aria-label={label} style={controlStyle} value={value}
      onChange={(event) => { onChange(equationIrChoice(event.target.value, options, label)); }}>
      {options.map((option) => <option key={option} value={option}>{option}</option>)}
    </select>
  </label>;
}

/**
 * Edit every serializable model/profile choice alongside the original playground source.
 *
 * @returns Saved compiler controls or catalogue-mode guidance.
 */
export default function EquationIrOptions() {
  const config = useStudioStore((state) => state.equationIrConfig);
  const setConfig = useStudioStore((state) => state.setEquationIrConfig);
  const sourceMode = useStudioStore((state) => state.sourceMode);
  const p = config.profile, m = config.model;

  /**
   * Change arithmetic settings and withdraw all outputs from the previous request.
   *
   * @param key - Arithmetic field.
   * @param value - Exact typed draft.
   */
  function profile<K extends keyof EquationIrProfileDraft>(key: K, value: EquationIrProfileDraft[K]): void {
    setConfig({ ...config, profile: { ...p, [key]: value } });
  }
  /**
   * Change model settings while preserving untouched declarations and draft text.
   *
   * @param key - Model field.
   * @param value - Exact typed draft.
   */
  function model<K extends keyof EquationIrModelDraft>(key: K, value: EquationIrModelDraft[K]): void {
    setConfig({ ...config, model: { ...m, [key]: value } });
  }

  return <details style={{ padding: "8px 12px", borderBottom: "1px solid var(--border)", maxHeight: "50%", overflow: "auto" }}>
    <summary style={{ cursor: "pointer" }}>Equation compiler settings</summary>
    {sourceMode !== "ode" ? <p>Choose ODE mode to compile the equation playground.</p> : <>
      <p style={{ color: "var(--text-secondary)" }}>
        Equations, initial values, parameters, threshold, reset and timestep come from the playground.
        These additional choices are saved with projects and sessions.
      </p>
      <fieldset style={{ marginBottom: 10 }}><legend>Arithmetic and registered outputs</legend>
        <div style={gridStyle}>
          {profileFields.map((field) => <TextSetting key={field.key} label={field.label}
            help={field.help} value={p[field.key]} onChange={(value) => { profile(field.key, value); }} />)}
          <ChoiceSetting label="Overflow" value={p.overflow} options={EQUATION_OVERFLOWS}
            onChange={(value) => { profile("overflow", value); }} />
          <ChoiceSetting label="Product rounding" value={p.rounding} options={EQUATION_ROUNDINGS}
            onChange={(value) => { profile("rounding", value); }} />
          <ChoiceSetting label="Function lowering" value={p.function_lowering} options={EQUATION_FUNCTION_PROFILES}
            onChange={(value) => { profile("function_lowering", value); }} />
          <ChoiceSetting label="Random precision" value={p.random_precision} options={EQUATION_RANDOM_PROFILES}
            onChange={(value) => { profile("random_precision", value); }} />
          <label><input type="checkbox" checked={p.signed}
            onChange={(event) => { profile("signed", event.target.checked); }} /> Signed scalar arithmetic
            <small style={{ display: "block", color: "var(--text-muted)" }}>
              The current equation datapath supports signed arithmetic. Unsigned requests refuse explicitly.
            </small>
          </label>
        </div>
      </fieldset>
      <fieldset style={{ marginBottom: 10 }}><legend>Integration, events and randomness</legend>
        <div style={gridStyle}>
          <ChoiceSetting label="Integrator" value={m.method} options={EQUATION_METHODS}
            onChange={(value) => { model("method", value); }} />
          <ChoiceSetting label="Event detection" value={m.detection} options={EQUATION_DETECTIONS}
            onChange={(value) => { model("detection", value); }} />
          <ChoiceSetting label="Threshold random generator" value={m.threshold_rng} options={EQUATION_THRESHOLD_GENERATORS}
            onChange={(value) => { model("threshold_rng", value); }} />
          {modelFields.map((field) => <TextSetting key={field.key} label={field.label}
            help={field.help} value={m[field.key]} onChange={(value) => { model(field.key, value); }} />)}
        </div>
      </fieldset>
      <fieldset><legend>Physical units</legend>
        <p>Strict mode validates dimensions. Unit declarations name every annotated value explicitly.</p>
        <div style={gridStyle}>
          <ChoiceSetting label="Units mode" value={m.units} options={EQUATION_UNIT_MODES}
            onChange={(value) => { model("units", value); }} />
          {unitFields.map((field) => <TextSetting key={field.key} label={field.label} multiline={field.multiline}
            help={field.help} value={m[field.key]} onChange={(value) => { model(field.key, value); }} />)}
        </div>
      </fieldset>
      <button type="button" onClick={() => { setConfig(defaultEquationIrConfiguration()); }}
        style={{ ...controlStyle, width: "auto", marginTop: 8, cursor: "pointer" }}>Reset compiler settings</button>
    </>}
  </details>;
}
