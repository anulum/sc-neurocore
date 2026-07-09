// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useEffect, useState } from "react";
import {
  evaluateDcls,
  fetchDclsBenchmark,
  fetchDclsInfo,
  type DclsBackendStatus,
  type DclsBenchmark,
  type DclsEvaluation,
  type DclsInfo,
} from "../api/client";
import BenchmarkContribution from "./BenchmarkContribution";

const BACKEND_BAR_COLOR: Record<string, string> = {
  rust: "#dea584",
  julia: "#9558b2",
  mojo: "#ff5f1f",
  go: "#00add8",
  python: "#80cbc4",
};

const Q88 = 256;

export function backendColor(b: DclsBackendStatus): string {
  if (!b.available) return "var(--text-muted)";
  if (!b.live) return "var(--accent)"; // declared, parity verified offline
  return b.bit_exact ? "var(--success)" : "var(--warning)";
}

export function backendLabel(b: DclsBackendStatus): string {
  if (!b.available) return "—";
  if (!b.live) return "offline ✓";
  return b.bit_exact ? "bit-exact" : "DIVERGES";
}

function TentChart({ gates, centre }: { gates: number[]; centre: number }) {
  const w = 260;
  const h = 90;
  const n = gates.length;
  const bw = w / n;
  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} style={{ display: "block" }}>
      <line x1={0} y1={h - 1} x2={w} y2={h - 1} stroke="var(--border)" />
      {gates.map((g, i) => {
        const bh = g * (h - 8);
        const isCentre = Math.round(centre) === i;
        return (
          <rect
            key={i}
            x={i * bw + 1}
            y={h - 1 - bh}
            width={bw - 2}
            height={bh}
            fill={isCentre ? "var(--accent)" : "var(--success)"}
            opacity={g > 0 ? 0.85 : 0.15}
          >
            <title>{`tap ${i}: gate ${g.toFixed(3)}`}</title>
          </rect>
        );
      })}
    </svg>
  );
}

export default function DclsPanel() {
  const [info, setInfo] = useState<DclsInfo | null>(null);
  const [benchmark, setBenchmark] = useState<DclsBenchmark | null>(null);
  const [evaluation, setEvaluation] = useState<DclsEvaluation | null>(null);
  const [centre, setCentre] = useState(3.0);
  const [sigma, setSigma] = useState(2.5);
  const [nTaps, setNTaps] = useState(12);

  useEffect(() => {
    void fetchDclsInfo().then(setInfo).catch(() => setInfo(null));
    void fetchDclsBenchmark().then(setBenchmark).catch(() => setBenchmark(null));
  }, []);

  useEffect(() => {
    const handle = window.setTimeout(() => {
      void evaluateDcls({
        centre_q88: Math.round(centre * Q88),
        sigma_q88: Math.max(1, Math.round(sigma * Q88)),
        n_taps: nTaps,
      })
        .then(setEvaluation)
        .catch(() => setEvaluation(null));
    }, 120);
    return () => window.clearTimeout(handle);
  }, [centre, sigma, nTaps]);

  const fwd = evaluation?.forward;
  const liveBitExact = fwd?.bit_exact ?? false;

  return (
    <div style={{ flex: 1, padding: 16, overflowY: "auto", fontSize: 12 }}>
      <h2 style={{ fontSize: 15, margin: "0 0 2px", color: "var(--accent)" }}>
        Learnable synaptic delays — DCLS-max tent kernel
      </h2>
      {info && (
        <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 12 }}>
          {info.provenance.authors.join(", ")} ({info.provenance.year}). {info.provenance.title}.{" "}
          <a href={`https://doi.org/${info.provenance.doi}`} target="_blank" rel="noreferrer"
            style={{ color: "var(--accent)" }}>
            {info.provenance.venue} · doi:{info.provenance.doi}
          </a>
        </div>
      )}

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        <div style={{ flex: "1 1 280px", minWidth: 260 }}>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>
            Learnable tent weighting over delay taps
          </div>
          {evaluation && (
            <TentChart gates={evaluation.profile.gates} centre={evaluation.profile.centre} />
          )}
          <div style={{ marginTop: 10, display: "grid", gap: 6 }}>
            <Slider label="centre (delay)" value={centre} min={0} max={nTaps - 1} step={0.05}
              onChange={setCentre} />
            <Slider label="sigma (half-width)" value={sigma} min={0.2} max={nTaps} step={0.05}
              onChange={setSigma} />
            <Slider label="delay taps" value={nTaps} min={3} max={32} step={1}
              onChange={(v) => setNTaps(Math.round(v))} />
          </div>
        </div>

        <div style={{ flex: "1 1 260px", minWidth: 240 }}>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>
            Q8.8 contraction — cross-backend parity
          </div>
          {fwd && (
            <>
              <div style={{
                padding: "4px 8px", marginBottom: 8, borderRadius: "var(--radius)",
                background: liveBitExact ? "rgba(129,199,132,0.12)" : "var(--bg-tertiary)",
                border: `1px solid ${liveBitExact ? "var(--success)" : "var(--border)"}`,
              }}>
                output = {fwd.reference_output.toFixed(3)} ({fwd.reference_output_q88} Q8.8)
                {" · "}{fwd.active_tap_count} active taps
                {fwd.overflow && <span style={{ color: "var(--warning)" }}> · overflow</span>}
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <tbody>
                  {fwd.backends.map((b) => (
                    <tr key={b.backend}>
                      <td style={{ padding: "1px 6px", fontFamily: "var(--font-mono)" }}>{b.backend}</td>
                      <td style={{ padding: "1px 6px", color: backendColor(b), fontWeight: 600 }}>
                        {backendLabel(b)}
                      </td>
                      <td style={{ padding: "1px 6px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                        {b.live && b.available ? b.output?.toFixed(3) : ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      </div>

      {benchmark && (
        <div style={{ marginTop: 16, paddingTop: 10, borderTop: "1px solid var(--border)" }}>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>
            Backend throughput — recorded {benchmark.workload.n_channels.toLocaleString()} channels
            × {benchmark.workload.n_taps} taps, speed-up over the Python floor
          </div>
          <div style={{ display: "grid", gap: 3 }}>
            {benchmark.backends.map((b) => {
              const max = benchmark.backends[0]?.speedup_over_python || 1;
              return (
                <div key={b.backend} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11 }}>
                  <span style={{ width: 56, fontFamily: "var(--font-mono)" }}>{b.backend}</span>
                  <div style={{ flex: 1, background: "var(--bg-tertiary)", borderRadius: 2, height: 14 }}>
                    <div style={{
                      width: `${(b.speedup_over_python / max) * 100}%`, height: "100%",
                      background: BACKEND_BAR_COLOR[b.backend] ?? "var(--accent)",
                      borderRadius: 2, minWidth: 2,
                    }} />
                  </div>
                  <span style={{ width: 96, textAlign: "right", fontFamily: "var(--font-mono)" }}>
                    {b.speedup_over_python.toFixed(1)}× · {b.median_call_ms.toFixed(2)}ms
                  </span>
                </div>
              );
            })}
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 5 }}>
            {benchmark.cpu} · {benchmark.isolation_mode} ·{" "}
            {benchmark.hardware_measurement_claimed ? "silicon" : "software measurement, not silicon"}
            {" · "}{benchmark.date_utc?.slice(0, 10)}
          </div>
        </div>
      )}

      <BenchmarkContribution />

      {info && (
        <div style={{
          marginTop: 16, paddingTop: 10, borderTop: "1px solid var(--border)",
          fontSize: 11, color: "var(--text-secondary)", display: "grid", gap: 3,
        }}>
          <div>
            <b>Fixed-point</b>: weights {info.fixed_point.weight_format}, accumulator{" "}
            {info.fixed_point.accumulator_format}, parity {info.fixed_point.parity}
          </div>
          <div>
            <b>RTL</b>: {info.rtl_modules.join(", ")} → {info.synthesis_target}
          </div>
          <div>
            <b>Backends</b>: {info.backend_order.join(" → ")} (fastest-first dispatch)
          </div>
        </div>
      )}
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11 }}>
      <span style={{ width: 110, color: "var(--text-secondary)" }}>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))} style={{ flex: 1 }} />
      <span style={{ width: 38, textAlign: "right", fontFamily: "var(--font-mono)" }}>
        {value.toFixed(2)}
      </span>
    </label>
  );
}
