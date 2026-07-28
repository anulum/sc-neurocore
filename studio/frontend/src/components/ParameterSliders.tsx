// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useStudioStore } from "../stores/studio";

function sliderRange(value: number): [number, number, number] {
  const absVal = Math.abs(value) || 1;
  return [value - absVal * 2, value + absVal * 2, absVal / 100];
}

function fmt(n: number): string {
  if (Math.abs(n) >= 100) return n.toFixed(1);
  if (Math.abs(n) >= 1) return n.toPrecision(4);
  if (n === 0) return "0";
  return n.toPrecision(3);
}

const PROTOCOLS = [
  { value: "constant", label: "Constant" },
  { value: "step", label: "Step (20%-80%)" },
  { value: "ramp", label: "Ramp (0 → I)" },
  { value: "pulse", label: "Pulse train" },
];

export function sliderBounds(
  value: number,
  range?: [number, number] | null,
): [number, number, number] {
  if (range && range[0] < range[1]) {
    const [lo, hi] = range;
    return [lo, hi, (hi - lo) / 200 || 1e-6];
  }
  return sliderRange(value);
}

function Slider({ label, value, onChange, min, max, step, unit, title }: {
  label: string; value: number;
  onChange: (v: number) => void;
  min?: number; max?: number; step?: number;
  unit?: string; title?: string;
}) {
  const [lo, hi, st] = min !== undefined
    ? [min, max!, step!]
    : sliderRange(value);
  return (
    <div className="slider-row" title={title}>
      <span className="slider-label">{label}</span>
      <input type="range" min={lo} max={hi} step={st} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))} />
      <span className="slider-value">
        {fmt(value)}
        {unit ? <span style={{ color: "var(--text-muted)", marginLeft: 3 }}>{unit}</span> : null}
      </span>
    </div>
  );
}

export default function ParameterSliders() {
  const {
    sourceMode, modelDetail, modelParams, setModelParam,
    modelIntegrator, modelQFormat, setModelIntegrator, setModelQFormat,
    odeParams, odeInit, setOdeParam, setOdeInit,
    current, dt, duration, protocol,
    setCurrent, setDt, setDuration, setProtocol,
  } = useStudioStore();

  return (
    <>
      {sourceMode === "model" && modelDetail && (
        <>
          <div className="panel-section">
            <div className="panel-header">
              Parameters ({modelDetail.params.length})
            </div>
            {modelDetail.params.map((p) => {
              const [lo, hi, st] = sliderBounds(
                modelParams[p.name] ?? p.default,
                p.range,
              );
              return (
                <Slider key={p.name} label={p.name}
                  value={modelParams[p.name] ?? p.default}
                  onChange={(v) => setModelParam(p.name, v)}
                  min={lo} max={hi} step={st}
                  unit={p.unit || undefined}
                  title={p.meaning || undefined} />
              );
            })}
          </div>
          {modelDetail.state_vars.length > 0 && (
            <div className="panel-section">
              <div className="panel-header">
                Initial State ({modelDetail.state_vars.length})
              </div>
              {modelDetail.state_vars.map((s) => (
                <Slider key={`init-${s.name}`} label={`${s.name}₀`}
                  value={modelParams[s.name] ?? s.default}
                  onChange={(v) => setModelParam(s.name, v)} />
              ))}
            </div>
          )}
          <div className="panel-section" data-testid="model-compile-configuration">
            <div className="panel-header">RTL Configuration</div>
            {modelDetail.compile_configuration ? (
              <>
                <div className="slider-row">
                  <span className="slider-label">integrator</span>
                  <select value={modelIntegrator} onChange={(event) => setModelIntegrator(event.target.value)}>
                    {modelDetail.compile_configuration.integrators.map((integrator) => (
                      <option key={integrator} value={integrator}>{integrator}</option>
                    ))}
                  </select>
                </div>
                <div className="slider-row">
                  <span className="slider-label">Q-format</span>
                  <select value={modelQFormat} onChange={(event) => setModelQFormat(event.target.value)}>
                    {modelDetail.compile_configuration.q_formats.map((qFormat) => (
                      <option key={qFormat} value={qFormat}>{qFormat}</option>
                    ))}
                  </select>
                </div>
              </>
            ) : (
              <span className="slider-value">No canonical schema-backed RTL path</span>
            )}
          </div>
        </>
      )}

      {sourceMode === "ode" && (
        <>
          <div className="panel-section">
            <div className="panel-header">Parameters</div>
            {Object.entries(odeParams).map(([key, value]) => (
              <Slider key={key} label={key} value={value}
                onChange={(v) => setOdeParam(key, v)} />
            ))}
          </div>
          <div className="panel-section">
            <div className="panel-header">Initial State</div>
            {Object.entries(odeInit).map(([key, value]) => (
              <Slider key={`init-${key}`} label={`${key}₀`} value={value}
                onChange={(v) => setOdeInit(key, v)} />
            ))}
          </div>
        </>
      )}

      <div className="panel-section">
        <div className="panel-header">Current Injection</div>
        <div className="slider-row">
          <span className="slider-label">protocol</span>
          <select value={protocol} onChange={(e) => setProtocol(e.target.value)}
            style={{ flex: 1 }}>
            {PROTOCOLS.map((p) => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
          </select>
        </div>
        <Slider label="I" value={current} onChange={setCurrent}
          min={-100} max={100} step={0.1} />
        <Slider label="dt" value={dt} onChange={setDt}
          min={0.001} max={1} step={0.001} />
        <Slider label="T (ms)" value={duration} onChange={setDuration}
          min={10} max={2000} step={10} />
      </div>
    </>
  );
}
