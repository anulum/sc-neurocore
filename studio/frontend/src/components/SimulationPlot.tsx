import { useRef, useEffect, useCallback, useState } from "react";
import { useStudioStore } from "../stores/studio";
import type {
  AnalysisResultMetadata,
  BifurcationResponse,
  CompareResponse,
  FICurveResponse,
  FreqResponse,
  HeatmapResponse,
  NullclineResponse,
  PrecisionResponse,
  SensitivityResponse,
} from "../api/client";
import { buildAnalysisEvidenceItems, buildSimulationEvidenceItems } from "../plotEvidence";
import EvidenceSummaryStrip from "./EvidenceSummaryStrip";

const COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ce93d8", "#90a4ae"];
const BG = "#0d1117";
const PANEL_BG = "#0a0e14";
const GRID = "#1a1f2a";
const AXIS = "#484f58";
const BORDER = "#21262d";

type AnalysisResult =
  | BifurcationResponse
  | CompareResponse
  | FICurveResponse
  | FreqResponse
  | HeatmapResponse
  | NullclineResponse
  | PrecisionResponse
  | SensitivityResponse
  | null;

function resultMetadata(result: AnalysisResult): AnalysisResultMetadata | null {
  return result?.analysis_metadata ?? null;
}

function niceStep(range: number, ticks: number): number {
  if (range <= 0 || !isFinite(range)) return 1;
  const rough = range / ticks;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const n = rough / mag;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
}

function drawAxes(
  ctx: CanvasRenderingContext2D,
  x0: number, y0: number, pw: number, ph: number,
  xMin: number, xMax: number, yMin: number, yMax: number,
  xLabel?: string,
) {
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  ctx.fillStyle = PANEL_BG;
  ctx.fillRect(x0, y0, pw, ph);
  ctx.strokeStyle = BORDER;
  ctx.lineWidth = 1;
  ctx.strokeRect(x0, y0, pw, ph);

  // Y ticks
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 0.5;
  ctx.font = "10px monospace";
  ctx.fillStyle = AXIS;
  ctx.textAlign = "right";
  const ys = niceStep(yRange, 4);
  for (let v = Math.ceil(yMin / ys) * ys; v <= yMax; v += ys) {
    const y = y0 + ph - ((v - yMin) / yRange) * ph;
    if (y < y0 + 2 || y > y0 + ph - 2) continue;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + pw, y); ctx.stroke();
    const lbl = Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3);
    ctx.fillText(lbl, x0 - 4, y + 3);
  }

  // X ticks
  ctx.textAlign = "center";
  const xs = niceStep(xRange, 6);
  for (let v = Math.ceil(xMin / xs) * xs; v <= xMax; v += xs) {
    const x = x0 + ((v - xMin) / xRange) * pw;
    ctx.beginPath(); ctx.strokeStyle = GRID; ctx.moveTo(x, y0); ctx.lineTo(x, y0 + ph); ctx.stroke();
    ctx.fillStyle = AXIS;
    ctx.fillText(v.toFixed(xs < 1 ? 2 : 0), x, y0 + ph + 12);
  }
  if (xLabel) {
    ctx.textAlign = "right";
    ctx.fillText(xLabel, x0 + pw, y0 + ph + 12);
  }
}

function drawLine(
  ctx: CanvasRenderingContext2D,
  x0: number, y0: number, pw: number, ph: number,
  xData: number[], yData: number[],
  xMin: number, xMax: number, yMin: number, yMax: number,
  color: string, lineWidth = 1.2,
) {
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  for (let i = 0; i < xData.length; i++) {
    const x = x0 + ((xData[i] - xMin) / xRange) * pw;
    const y = y0 + ph - ((yData[i] - yMin) / yRange) * ph;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

export default function SimulationPlot() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef({ xMin: NaN, xMax: NaN, yMin: NaN, yMax: NaN });
  const dragRef = useRef<{ startX: number; startY: number; origXMin: number; origXMax: number; origYMin: number; origYMax: number } | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const crosshairRef = useRef<number | null>(null);
  const store = useStudioStore();
  const {
    result, activeTab, fiResult, bifResult, sensResult, precResult,
    heatmapResult, compareResult, nullclineResult, freqResult, staResult,
    charResult, multiResults, importedTrace, networkResult,
  } = store;
  const analysisMetadata =
    activeTab === "fi-curve" ? resultMetadata(fiResult) :
    activeTab === "bifurcation" ? resultMetadata(bifResult) :
    activeTab === "sensitivity" ? resultMetadata(sensResult) :
    activeTab === "precision" ? resultMetadata(precResult) :
    activeTab === "heatmap" ? resultMetadata(heatmapResult) :
    activeTab === "compare" ? resultMetadata(compareResult) :
    activeTab === "freq" ? resultMetadata(freqResult) :
    activeTab === "phase" ? resultMetadata(nullclineResult) :
    null;
  const simulationMetadata = activeTab === "trace" ? result?.run_metadata ?? null : null;

  function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (activeTab !== "heatmap" || !heatmapResult) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const x = (e.clientX - rect.left) * dpr;
    const y = (e.clientY - rect.top) * dpr;
    const L = 52, T = 8, R = 12, B = 18;
    const pw = canvas.width - L * dpr - R * dpr;
    const ph = canvas.height - T * dpr - B * dpr - 16 * dpr;
    const { x_values, y_values } = heatmapResult;
    const xi = Math.floor(((x - L * dpr) / pw) * x_values.length);
    const yi = y_values.length - 1 - Math.floor(((y - T * dpr) / ph) * y_values.length);
    if (xi >= 0 && xi < x_values.length && yi >= 0 && yi < y_values.length) {
      const params = store.sourceMode === "model" ? { ...store.modelParams } : { ...store.odeParams };
      params[heatmapResult.param_x] = x_values[xi];
      params[heatmapResult.param_y] = y_values[yi];
      if (store.sourceMode === "model") {
        useStudioStore.setState({ modelParams: params, activeTab: "trace" });
      } else {
        useStudioStore.setState({ odeParams: params, activeTab: "trace" });
      }
      store.runSimulation();
    }
  }

  function handleWheel(e: React.WheelEvent) {
    if (activeTab !== "trace" || !result) return;
    e.preventDefault();
    const z = zoomRef.current;
    const time = result.time;
    if (isNaN(z.xMin)) { z.xMin = time[0]; z.xMax = time[time.length - 1]; }
    const range = z.xMax - z.xMin;
    const factor = e.deltaY > 0 ? 1.2 : 0.8;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) / rect.width;
    const center = z.xMin + range * mouseX;
    const newRange = range * factor;
    z.xMin = center - newRange * mouseX;
    z.xMax = center + newRange * (1 - mouseX);
    draw();
  }

  function handleMouseDown(e: React.MouseEvent) {
    if (activeTab !== "trace" || !result) return;
    const z = zoomRef.current;
    if (isNaN(z.xMin)) return;
    dragRef.current = { startX: e.clientX, startY: e.clientY, origXMin: z.xMin, origXMax: z.xMax, origYMin: z.yMin, origYMax: z.yMax };
  }

  function handleMouseMove(e: React.MouseEvent) {
    const d = dragRef.current;
    if (d && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      const dx = (e.clientX - d.startX) / rect.width;
      const xRange = d.origXMax - d.origXMin;
      zoomRef.current.xMin = d.origXMin - dx * xRange;
      zoomRef.current.xMax = d.origXMax - dx * xRange;
      setTooltip(null);
      draw();
      return;
    }
    // Tooltip on trace view
    if (activeTab === "trace" && result && canvasRef.current) {
      const rect = canvasRef.current.getBoundingClientRect();
      const L = 52, pw = rect.width - L - 12;
      const fracX = (e.clientX - rect.left - L) / pw;
      if (fracX < 0 || fracX > 1) { setTooltip(null); return; }
      const z = zoomRef.current;
      const t0 = isNaN(z.xMin) ? result.time[0] : z.xMin;
      const t1 = isNaN(z.xMax) ? result.time[result.time.length - 1] : z.xMax;
      const tAt = t0 + fracX * (t1 - t0);
      const idx = Math.round(tAt / result.dt);
      const vars = Object.keys(result.states);
      const vals = vars.map((v) => {
        const arr = result.states[v];
        const i = Math.min(Math.max(idx, 0), arr.length - 1);
        return `${v}=${arr[i].toFixed(2)}`;
      }).join(" ");
      crosshairRef.current = e.clientX - rect.left;
      setTooltip({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        text: `t=${tAt.toFixed(1)} ${vals}`,
      });
      draw();
    } else {
      crosshairRef.current = null;
      setTooltip(null);
    }
  }

  function handleMouseUp() { dragRef.current = null; }
  function handleMouseLeave() { dragRef.current = null; crosshairRef.current = null; setTooltip(null); draw(); }

  function resetZoom() {
    zoomRef.current = { xMin: NaN, xMax: NaN, yMin: NaN, yMax: NaN };
    draw();
  }

  // Reset zoom when result changes
  useEffect(() => {
    zoomRef.current = { xMin: NaN, xMax: NaN, yMin: NaN, yMax: NaN };
  }, [result]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);
    if (w < 100 || h < 100) return;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);

    const L = 52, R = 12, T = 8, B = 18;
    const pw = w - L - R;

    // f-I curve view
    if (activeTab === "fi-curve" && fiResult) {
      const ph = h - T - B;
      const xMin = fiResult.currents[0], xMax = fiResult.currents[fiResult.currents.length - 1];
      let yMax = Math.max(...fiResult.rates, 1);
      drawAxes(ctx, L, T, pw, ph, xMin, xMax, 0, yMax * 1.1, "I (nA)");
      drawLine(ctx, L, T, pw, ph, fiResult.currents, fiResult.rates, xMin, xMax, 0, yMax * 1.1, "#4fc3f7", 2);
      ctx.fillStyle = AXIS;
      ctx.font = "10px monospace";
      ctx.textAlign = "left";
      ctx.fillText("f (Hz)", L + 4, T + 12);
      return;
    }

    if (!result) {
      ctx.fillStyle = AXIS;
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Select a model and adjust parameters", w / 2, h / 2);
      return;
    }

    const time = result.time;
    const vars = Object.keys(result.states);
    const tMin = time[0], tMax = time[time.length - 1];
    const hasSpikes = result.spikes.length > 0;

    // Phase portrait (2+ variables)
    if (activeTab === "phase" && vars.length >= 2) {
      const ph = h - T - B;
      const xData = result.states[vars[0]], yData = result.states[vars[1]];
      let xMin = Math.min(...xData), xMax = Math.max(...xData);
      let yMin = Math.min(...yData), yMax = Math.max(...yData);
      const xPad = (xMax - xMin) * 0.05 || 1;
      const yPad = (yMax - yMin) * 0.05 || 1;
      xMin -= xPad; xMax += xPad; yMin -= yPad; yMax += yPad;

      drawAxes(ctx, L, T, pw, ph, xMin, xMax, yMin, yMax, vars[0]);
      // Draw trajectory with fading colour
      for (let i = 1; i < xData.length; i++) {
        const alpha = 0.15 + 0.85 * (i / xData.length);
        ctx.strokeStyle = `rgba(79, 195, 247, ${alpha})`;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(
          L + ((xData[i - 1] - xMin) / (xMax - xMin)) * pw,
          T + ph - ((yData[i - 1] - yMin) / (yMax - yMin)) * ph
        );
        ctx.lineTo(
          L + ((xData[i] - xMin) / (xMax - xMin)) * pw,
          T + ph - ((yData[i] - yMin) / (yMax - yMin)) * ph
        );
        ctx.stroke();
      }
      // Start and end markers
      const sx = L + ((xData[0] - xMin) / (xMax - xMin)) * pw;
      const sy = T + ph - ((yData[0] - yMin) / (yMax - yMin)) * ph;
      ctx.fillStyle = "#81c784";
      ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "left";
      ctx.fillText(vars[1], L + 4, T + 12);

      // Nullcline overlay
      if (nullclineResult) {
        const xRange = xMax - xMin || 1;
        const yRange = yMax - yMin || 1;
        for (const [nc, color] of [
          [nullclineResult.nullcline_0, "#ff5252"],
          [nullclineResult.nullcline_1, "#81c784"],
        ] as const) {
          ctx.fillStyle = color;
          for (const [px, py] of nc.points) {
            const cx = L + ((px - xMin) / xRange) * pw;
            const cy = T + ph - ((py - yMin) / yRange) * ph;
            if (cx >= L && cx <= L + pw && cy >= T && cy <= T + ph) {
              ctx.fillRect(cx - 1, cy - 1, 2, 2);
            }
          }
        }
        ctx.font = "9px monospace"; ctx.textAlign = "right";
        ctx.fillStyle = "#ff5252"; ctx.fillText(`d${vars[0]}/dt=0`, L + pw - 4, T + ph - 16);
        ctx.fillStyle = "#81c784"; ctx.fillText(`d${vars[1]}/dt=0`, L + pw - 4, T + ph - 4);
      }
      return;
    }

    // ISI histogram
    if (activeTab === "isi" && result.stats.isi_histogram) {
      const ph = h - T - B;
      const hist = result.stats.isi_histogram as { counts: number[]; edges: number[] };
      const maxCount = Math.max(...hist.counts, 1);
      const xMin = hist.edges[0], xMax = hist.edges[hist.edges.length - 1];

      drawAxes(ctx, L, T, pw, ph, xMin, xMax, 0, maxCount * 1.1, "ISI (ms)");
      ctx.fillStyle = "rgba(79, 195, 247, 0.6)";
      const xRange = xMax - xMin || 1;
      for (let i = 0; i < hist.counts.length; i++) {
        const bx = L + ((hist.edges[i] - xMin) / xRange) * pw;
        const bw = ((hist.edges[i + 1] - hist.edges[i]) / xRange) * pw;
        const bh = (hist.counts[i] / (maxCount * 1.1)) * ph;
        ctx.fillRect(bx, T + ph - bh, Math.max(bw - 1, 1), bh);
      }
      ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "left";
      ctx.fillText("count", L + 4, T + 12);
      return;
    }

    // Bifurcation diagram (#2)
    if (activeTab === "bifurcation" && bifResult) {
      const ph = h - T - B;
      const { param_values, attractors } = bifResult;
      const xMin = param_values[0], xMax = param_values[param_values.length - 1];
      let yMin = Infinity, yMax = -Infinity;
      for (const a of attractors) for (const v of a) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
      if (!isFinite(yMin)) { yMin = -80; yMax = 40; }
      const yPad = (yMax - yMin) * 0.05 || 1;
      yMin -= yPad; yMax += yPad;
      drawAxes(ctx, L, T, pw, ph, xMin, xMax, yMin, yMax, bifResult.param_name);
      ctx.fillStyle = "rgba(79,195,247,0.5)";
      for (let i = 0; i < param_values.length; i++) {
        const x = L + ((param_values[i] - xMin) / (xMax - xMin || 1)) * pw;
        for (const v of attractors[i]) {
          const y = T + ph - ((v - yMin) / (yMax - yMin)) * ph;
          ctx.fillRect(x - 1, y - 1, 2, 2);
        }
      }
      ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "left";
      ctx.fillText("V attractor", L + 4, T + 12);
      return;
    }

    // 2D Heatmap
    if (activeTab === "heatmap" && heatmapResult) {
      const ph = h - T - B - 16;
      const { x_values, y_values, rates, rate_min, rate_max } = heatmapResult;
      const xMin = x_values[0], xMax = x_values[x_values.length - 1];
      const yMin = y_values[0], yMax = y_values[y_values.length - 1];
      const rRange = rate_max - rate_min || 1;

      drawAxes(ctx, L, T, pw, ph, xMin, xMax, yMin, yMax, heatmapResult.param_x);
      const cellW = pw / x_values.length;
      const cellH = ph / y_values.length;
      for (let j = 0; j < y_values.length; j++) {
        for (let i = 0; i < x_values.length; i++) {
          const norm = (rates[j][i] - rate_min) / rRange;
          const r = Math.floor(norm * 200 + 20);
          const g = Math.floor(norm * 50);
          const b = Math.floor((1 - norm) * 200 + 55);
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          const cx = L + (i / x_values.length) * pw;
          const cy = T + ph - ((j + 1) / y_values.length) * ph;
          ctx.fillRect(cx, cy, cellW + 1, cellH + 1);
        }
      }
      ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "left";
      ctx.fillText(`${heatmapResult.param_y} vs ${heatmapResult.param_x}  (${rate_min.toFixed(0)}–${rate_max.toFixed(0)} Hz)`, L + 4, T + 12);
      return;
    }

    // Sensitivity (#8)
    if (activeTab === "sensitivity" && sensResult) {
      const ph = h - T - B - 20;
      const sens = sensResult.sensitivities.slice(0, 15);
      if (sens.length === 0) return;
      const maxS = Math.max(...sens.map((s) => s.sensitivity), 0.01);
      const barH = Math.min(20, ph / sens.length - 2);
      ctx.font = "10px monospace";
      sens.forEach((s, i) => {
        const y = T + i * (barH + 2);
        const bw = (s.sensitivity / maxS) * (pw - 80);
        ctx.fillStyle = "rgba(79,195,247,0.6)";
        ctx.fillRect(L + 70, y, bw, barH);
        ctx.fillStyle = AXIS; ctx.textAlign = "right";
        ctx.fillText(s.param, L + 65, y + barH - 4);
        ctx.textAlign = "left";
        ctx.fillText(s.sensitivity.toFixed(3), L + 75 + bw, y + barH - 4);
      });
      ctx.fillStyle = AXIS; ctx.textAlign = "left";
      ctx.fillText(`base rate: ${sensResult.base_rate} Hz`, L + 4, h - 8);
      return;
    }

    // Precision compare (#5)
    if (activeTab === "precision" && precResult) {
      const ph = (h - T - B - 30) / 2;
      const float_v = precResult.float_result.states[precResult.error.variable];
      const fixed_v = precResult.fixed_result.states[precResult.error.variable];
      const time_f = precResult.float_result.time;
      const tMin = time_f[0], tMax = time_f[time_f.length - 1];
      let vMin = Math.min(...float_v, ...fixed_v);
      let vMax = Math.max(...float_v, ...fixed_v);
      const vPad = (vMax - vMin) * 0.05 || 1;
      vMin -= vPad; vMax += vPad;

      drawAxes(ctx, L, T, pw, ph, tMin, tMax, vMin, vMax);
      drawLine(ctx, L, T, pw, ph, time_f, float_v, tMin, tMax, vMin, vMax, "#4fc3f7", 1.2);
      drawLine(ctx, L, T, pw, ph, time_f, fixed_v, tMin, tMax, vMin, vMax, "#ff5252", 1.2);
      ctx.font = "10px monospace"; ctx.textAlign = "left";
      ctx.fillStyle = "#4fc3f7"; ctx.fillText("float64", L + 6, T + 12);
      ctx.fillStyle = "#ff5252"; ctx.fillText("Q8.8", L + 60, T + 12);

      // Error trace
      const errY = T + ph + 16;
      const errH = ph - 8;
      const errMax = Math.max(...precResult.error.trace, 0.001);
      drawAxes(ctx, L, errY, pw, errH, tMin, tMax, 0, errMax * 1.1, "ms");
      drawLine(ctx, L, errY, pw, errH, time_f, precResult.error.trace, tMin, tMax, 0, errMax * 1.1, "#ffb74d", 1.5);
      ctx.fillStyle = "#ffb74d"; ctx.font = "10px monospace"; ctx.textAlign = "left";
      ctx.fillText(`error (max=${precResult.error.max_error.toFixed(4)}, rms=${precResult.error.rms_error.toFixed(4)})`, L + 6, errY + 12);
      return;
    }

    // Comparison view
    if (activeTab === "compare" && compareResult) {
      const ph = (h - T - B - 10) / 2;
      for (const [idx, label, res] of [[0, "A", compareResult.a], [1, "B", compareResult.b]] as const) {
        const yOff = T + idx * (ph + 10);
        const v0 = Object.keys(res.states)[0];
        const data = res.states[v0];
        const tm = res.time;
        let yMin = Math.min(...data), yMax = Math.max(...data);
        const yPad = (yMax - yMin) * 0.05 || 1;
        yMin -= yPad; yMax += yPad;
        drawAxes(ctx, L, yOff, pw, ph, tm[0], tm[tm.length - 1], yMin, yMax);
        drawLine(ctx, L, yOff, pw, ph, tm, data, tm[0], tm[tm.length - 1], yMin, yMax, COLORS[idx], 1.2);
        ctx.fillStyle = COLORS[idx]; ctx.font = "10px monospace"; ctx.textAlign = "left";
        ctx.fillText(`${label}: ${res.model_name || "custom"} (${res.stats.rate_hz} Hz)`, L + 6, yOff + 12);
      }
      return;
    }

    // Frequency response
    if (activeTab === "freq" && freqResult) {
      const ph = h - T - B;
      const xMin = freqResult.frequencies_hz[0];
      const xMax = freqResult.frequencies_hz[freqResult.frequencies_hz.length - 1];
      const yMax = Math.max(...freqResult.rates, 1);
      drawAxes(ctx, L, T, pw, ph, xMin, xMax, 0, yMax * 1.1, "freq (Hz)");
      drawLine(ctx, L, T, pw, ph, freqResult.frequencies_hz, freqResult.rates,
        xMin, xMax, 0, yMax * 1.1, "#4fc3f7", 2);
      ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "left";
      ctx.fillText(`rate (Hz) @ amplitude=${freqResult.amplitude}`, L + 4, T + 12);
      return;
    }

    // Spike-triggered average
    if (activeTab === "sta" && staResult && staResult.time_ms.length > 0) {
      const ph = h - T - B;
      const xMin = staResult.time_ms[0], xMax = staResult.time_ms[staResult.time_ms.length - 1];
      let yMin = Math.min(...staResult.average), yMax = Math.max(...staResult.average);
      const yPad = (yMax - yMin) * 0.05 || 1;
      yMin -= yPad; yMax += yPad;
      drawAxes(ctx, L, T, pw, ph, xMin, xMax, yMin, yMax, "ms (relative to spike)");
      drawLine(ctx, L, T, pw, ph, staResult.time_ms, staResult.average, xMin, xMax, yMin, yMax, "#4fc3f7", 2);
      // Vertical line at t=0
      const x0 = L + ((0 - xMin) / (xMax - xMin)) * pw;
      ctx.strokeStyle = "#ff5252"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x0, T); ctx.lineTo(x0, T + ph); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "left";
      ctx.fillText(`STA (n=${staResult.n_spikes} spikes)`, L + 4, T + 12);
      return;
    }

    // Characterize dashboard
    if (activeTab === "characterize" && charResult) {
      ctx.fillStyle = "#e6edf3"; ctx.font = "12px sans-serif"; ctx.textAlign = "left";
      let y = T + 16;
      const lineH = 18;
      const col1 = L, col2 = L + pw / 2;

      ctx.fillStyle = "#4fc3f7"; ctx.font = "bold 13px sans-serif";
      ctx.fillText("Model Characterisation", col1, y); y += lineH + 4;

      ctx.font = "11px monospace"; ctx.fillStyle = "#e6edf3";
      ctx.fillText(`Pattern: ${charResult.pattern.description}`, col1, y); y += lineH;
      ctx.fillText(`Threshold current: ${charResult.threshold_current ?? "N/A"} nA`, col1, y); y += lineH;
      ctx.fillText(`Max firing rate: ${charResult.max_rate} Hz`, col1, y); y += lineH;
      ctx.fillText(`Spikes: ${charResult.spike_count}`, col1, y); y += lineH;
      if (charResult.stats.isi_mean_ms) {
        ctx.fillText(`ISI: ${charResult.stats.isi_mean_ms} ms (CV=${charResult.stats.isi_cv})`, col1, y); y += lineH;
      }

      y += 8;
      ctx.fillStyle = "#4fc3f7"; ctx.font = "bold 11px sans-serif";
      ctx.fillText("State Variable Ranges", col1, y); y += lineH;
      ctx.font = "10px monospace"; ctx.fillStyle = "#8b949e";
      for (const [v, r] of Object.entries(charResult.state_ranges)) {
        ctx.fillText(`${v}: [${r.min}, ${r.max}] mean=${r.mean}`, col1, y); y += lineH - 2;
      }

      y += 8;
      ctx.fillStyle = "#4fc3f7"; ctx.font = "bold 11px sans-serif";
      ctx.fillText("Top Sensitive Parameters", col1, y); y += lineH;
      ctx.font = "10px monospace"; ctx.fillStyle = "#8b949e";
      for (const s of charResult.top_sensitivities) {
        ctx.fillText(`${s.param}: ±${s.rate_change} Hz`, col1, y); y += lineH - 2;
      }

      // f-I curve in right half
      const fiX = col2, fiY = T + 20, fiW = pw / 2 - 20, fiH = h - T - B - 40;
      const curs = charResult.fi_curve.currents;
      const rts = charResult.fi_curve.rates;
      const rMax = Math.max(...rts, 1);
      drawAxes(ctx, fiX, fiY, fiW, fiH, curs[0], curs[curs.length - 1], 0, rMax * 1.1, "I (nA)");
      drawLine(ctx, fiX, fiY, fiW, fiH, curs, rts, curs[0], curs[curs.length - 1], 0, rMax * 1.1, "#4fc3f7", 2);
      ctx.fillStyle = "#4fc3f7"; ctx.font = "10px monospace"; ctx.textAlign = "left";
      ctx.fillText("f-I curve", fiX + 4, fiY + 12);
      return;
    }

    // Multi-model overlay
    if (activeTab === "multi" && multiResults && multiResults.length > 0) {
      const ph = h - T - B;
      let tMin = Infinity, tMax = -Infinity, vMin = Infinity, vMax = -Infinity;
      for (const r of multiResults) {
        if (r.time[0] < tMin) tMin = r.time[0];
        if (r.time[r.time.length - 1] > tMax) tMax = r.time[r.time.length - 1];
        const v0 = Object.keys(r.states)[0];
        for (const v of r.states[v0]) {
          if (isFinite(v)) { if (v < vMin) vMin = v; if (v > vMax) vMax = v; }
        }
      }
      const vPad = (vMax - vMin) * 0.06 || 1;
      vMin -= vPad; vMax += vPad;
      drawAxes(ctx, L, T, pw, ph, tMin, tMax, vMin, vMax, "ms");
      multiResults.forEach((r, i) => {
        const v0 = Object.keys(r.states)[0];
        drawLine(ctx, L, T, pw, ph, r.time, r.states[v0], tMin, tMax, vMin, vMax, COLORS[i % COLORS.length], 1.5);
      });
      ctx.font = "10px monospace";
      multiResults.forEach((r, i) => {
        const name = r.model_name || `Model ${i + 1}`;
        ctx.fillStyle = COLORS[i % COLORS.length];
        ctx.fillRect(L + 6 + i * 120, T + 4, 8, 2);
        ctx.textAlign = "left";
        ctx.fillText(`${name} (${r.stats.rate_hz}Hz)`, L + 17 + i * 120, T + 9);
      });
      return;
    }

    // Network E-I raster + rates
    if (activeTab === "network" && networkResult) {
      const rasterH = Math.floor((h - T - B) * 0.6);
      const rateH = h - T - B - rasterH - 10;

      // Raster plot
      ctx.fillStyle = PANEL_BG; ctx.fillRect(L, T, pw, rasterH);
      ctx.strokeStyle = BORDER; ctx.strokeRect(L, T, pw, rasterH);
      const dur = networkResult.duration;
      for (let i = 0; i < networkResult.spike_times.length; i++) {
        const t = networkResult.spike_times[i];
        const n = networkResult.spike_neurons[i];
        const x = L + (t / dur) * pw;
        const y = T + (n / networkResult.n_total) * rasterH;
        ctx.fillStyle = n < networkResult.n_exc ? "#4fc3f7" : "#ff5252";
        ctx.fillRect(x, y, 1.5, 1.5);
      }
      ctx.fillStyle = "#4fc3f7"; ctx.font = "9px monospace"; ctx.textAlign = "left";
      ctx.fillText(`E (${networkResult.n_exc})`, L + 4, T + 10);
      ctx.fillStyle = "#ff5252";
      ctx.fillText(`I (${networkResult.n_inh})`, L + 60, T + 10);
      ctx.fillStyle = AXIS;
      ctx.fillText(`${networkResult.n_spikes} spikes`, L + 120, T + 10);

      // Population rates
      const rateY = T + rasterH + 10;
      const rt = networkResult.rate_time;
      if (rt.length > 1) {
        const rMax = Math.max(...networkResult.exc_rates, ...networkResult.inh_rates, 1);
        drawAxes(ctx, L, rateY, pw, rateH, rt[0], rt[rt.length - 1], 0, rMax * 1.1, "ms");
        drawLine(ctx, L, rateY, pw, rateH, rt, networkResult.exc_rates, rt[0], rt[rt.length - 1], 0, rMax * 1.1, "#4fc3f7", 1.5);
        drawLine(ctx, L, rateY, pw, rateH, rt, networkResult.inh_rates, rt[0], rt[rt.length - 1], 0, rMax * 1.1, "#ff5252", 1.5);
        ctx.fillStyle = AXIS; ctx.font = "9px monospace"; ctx.textAlign = "left";
        ctx.fillText(`E: ${networkResult.mean_exc_rate}Hz  I: ${networkResult.mean_inh_rate}Hz`, L + 4, rateY + 10);
      }
      return;
    }

    // Default: Trace view (with nullcline overlay on phase + imported trace overlay)
    // Apply zoom viewport if set
    const z = zoomRef.current;
    const zTMin = isNaN(z.xMin) ? tMin : z.xMin;
    const zTMax = isNaN(z.xMax) ? tMax : z.xMax;

    // Layout: voltage 65%, current 15%, raster 8%, x-labels
    const gap = 4;
    const rasterH = hasSpikes ? 22 : 0;
    const currentH = 40;
    const xLabelH = 16;
    const voltH = h - T - currentH - rasterH - gap * 2 - xLabelH;
    if (voltH < 30) return;

    // Compute Y range
    let vMin = Infinity, vMax = -Infinity;
    for (const v of vars) {
      for (const val of result.states[v]) {
        if (isFinite(val)) { if (val < vMin) vMin = val; if (val > vMax) vMax = val; }
      }
    }
    const vPad = (vMax - vMin) * 0.06 || 1;
    vMin -= vPad; vMax += vPad;

    // Voltage plot
    drawAxes(ctx, L, T, pw, voltH, zTMin, zTMax, vMin, vMax);
    vars.forEach((v, i) => {
      drawLine(ctx, L, T, pw, voltH, time, result.states[v], zTMin, zTMax, vMin, vMax, COLORS[i % COLORS.length]);
    });
    // Y-axis label
    ctx.save();
    ctx.translate(10, T + voltH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = AXIS; ctx.font = "9px monospace"; ctx.textAlign = "center";
    ctx.fillText("mV", 0, 0);
    ctx.restore();
    // Spike markers
    if (hasSpikes) {
      ctx.strokeStyle = "rgba(255,82,82,0.2)"; ctx.lineWidth = 1;
      for (const idx of result.spikes) {
        const x = L + ((idx * result.dt - zTMin) / (zTMax - zTMin || 1)) * pw;
        ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + voltH); ctx.stroke();
      }
    }
    // Legend
    ctx.font = "10px monospace";
    vars.forEach((v, i) => {
      ctx.fillStyle = COLORS[i % COLORS.length];
      ctx.fillRect(L + 6 + i * 52, T + 4, 8, 2);
      ctx.textAlign = "left"; ctx.fillText(v, L + 17 + i * 52, T + 9);
    });

    // Imported trace overlay
    if (importedTrace) {
      ctx.setLineDash([4, 3]);
      drawLine(ctx, L, T, pw, voltH, importedTrace.time, importedTrace.voltage,
        zTMin, zTMax, vMin, vMax, "#ff9800", 1.5);
      ctx.setLineDash([]);
      ctx.fillStyle = "#ff9800"; ctx.font = "9px monospace"; ctx.textAlign = "left";
      ctx.fillText("imported", L + 6 + vars.length * 52, T + 9);
    }

    // Current plot
    const curY = T + voltH + gap;
    const I = result.current_trace;
    let iMin = Math.min(...I), iMax = Math.max(...I);
    if (iMin === iMax) { iMin -= 1; iMax += 1; }
    drawAxes(ctx, L, curY, pw, currentH, zTMin, zTMax, iMin, iMax * 1.1);
    drawLine(ctx, L, curY, pw, currentH, time, I, zTMin, zTMax, iMin, iMax * 1.1, "#ffb74d", 1.5);
    ctx.fillStyle = "#ffb74d"; ctx.font = "10px monospace"; ctx.textAlign = "left";
    ctx.fillText("I", L + 4, curY + 10);
    // Y-axis label for current
    ctx.save();
    ctx.translate(10, curY + currentH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = AXIS; ctx.font = "9px monospace"; ctx.textAlign = "center";
    ctx.fillText("nA", 0, 0);
    ctx.restore();

    // Spike raster
    if (hasSpikes) {
      const rasY = curY + currentH + gap;
      ctx.fillStyle = PANEL_BG; ctx.fillRect(L, rasY, pw, rasterH);
      ctx.strokeStyle = BORDER; ctx.lineWidth = 1; ctx.strokeRect(L, rasY, pw, rasterH);
      ctx.strokeStyle = "#ff5252"; ctx.lineWidth = 1.5;
      for (const idx of result.spikes) {
        const x = L + ((idx * result.dt - zTMin) / (zTMax - zTMin || 1)) * pw;
        ctx.beginPath(); ctx.moveTo(x, rasY + 2); ctx.lineTo(x, rasY + rasterH - 2); ctx.stroke();
      }
    }

    // X-axis labels
    ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "center";
    const xs = niceStep(zTMax - zTMin, 6);
    for (let v = Math.ceil(zTMin / xs) * xs; v <= zTMax; v += xs) {
      const x = L + ((v - zTMin) / (zTMax - zTMin || 1)) * pw;
      ctx.fillText(v.toFixed(0), x, h - 2);
    }
    ctx.textAlign = "right"; ctx.fillText("ms", L + pw, h - 2);

    // Crosshair
    if (crosshairRef.current !== null) {
      const cx = crosshairRef.current;
      ctx.strokeStyle = "rgba(79,195,247,0.3)"; ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.moveTo(cx, T); ctx.lineTo(cx, h - 10); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Zoom indicator
    if (!isNaN(zoomRef.current.xMin)) {
      ctx.fillStyle = "#4fc3f7"; ctx.font = "9px monospace"; ctx.textAlign = "right";
      ctx.fillText(`zoom: ${zTMin.toFixed(1)}–${zTMax.toFixed(1)} ms (dbl-click to reset)`, L + pw - 2, h - 2);
    }
  }, [result, activeTab, fiResult, bifResult, sensResult, precResult, heatmapResult, compareResult, nullclineResult, freqResult, staResult, charResult, multiResults, importedTrace, networkResult]);

  useEffect(() => {
    draw();
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [draw]);

  return (
    <div ref={containerRef} style={{
      flex: 1, position: "relative", overflow: "hidden",
    }}>
      {tooltip && (
        <div style={{
          position: "absolute", left: tooltip.x + 10, top: tooltip.y - 24,
          background: "rgba(22,27,34,0.95)", color: "#e6edf3",
          padding: "2px 6px", borderRadius: 3, fontSize: 10,
          fontFamily: "var(--font-mono)", pointerEvents: "none",
          border: "1px solid var(--border)", whiteSpace: "nowrap",
        }}>{tooltip.text}</div>
      )}
      {analysisMetadata && (
        <EvidenceSummaryStrip variant="overlay" items={buildAnalysisEvidenceItems(analysisMetadata)} />
      )}
      {simulationMetadata && (
        <EvidenceSummaryStrip variant="overlay" items={buildSimulationEvidenceItems(simulationMetadata)} />
      )}
      <canvas ref={canvasRef}
        onClick={handleCanvasClick}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onDoubleClick={resetZoom}
        style={{
          position: "absolute", top: 0, left: 0, width: "100%", height: "100%",
          cursor: activeTab === "heatmap" ? "crosshair" :
                  activeTab === "trace" ? "grab" : "default",
        }} />
    </div>
  );
}
