import { useRef, useEffect, useCallback } from "react";
import { useStudioStore } from "../stores/studio";

const COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ce93d8", "#90a4ae"];
const BG = "#0d1117";
const PANEL_BG = "#0a0e14";
const GRID = "#1a1f2a";
const AXIS = "#484f58";
const BORDER = "#21262d";

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
  const { result, activeTab, fiResult, bifResult, sensResult, precResult, heatmapResult } = useStudioStore();

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

    // Default: Trace view
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
    drawAxes(ctx, L, T, pw, voltH, tMin, tMax, vMin, vMax);
    vars.forEach((v, i) => {
      drawLine(ctx, L, T, pw, voltH, time, result.states[v], tMin, tMax, vMin, vMax, COLORS[i % COLORS.length]);
    });
    // Spike markers
    if (hasSpikes) {
      ctx.strokeStyle = "rgba(255,82,82,0.2)"; ctx.lineWidth = 1;
      for (const idx of result.spikes) {
        const x = L + ((idx * result.dt - tMin) / (tMax - tMin || 1)) * pw;
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

    // Current plot
    const curY = T + voltH + gap;
    const I = result.current_trace;
    let iMin = Math.min(...I), iMax = Math.max(...I);
    if (iMin === iMax) { iMin -= 1; iMax += 1; }
    drawAxes(ctx, L, curY, pw, currentH, tMin, tMax, iMin, iMax * 1.1);
    drawLine(ctx, L, curY, pw, currentH, time, I, tMin, tMax, iMin, iMax * 1.1, "#ffb74d", 1.5);
    ctx.fillStyle = "#ffb74d"; ctx.font = "10px monospace"; ctx.textAlign = "left";
    ctx.fillText("I", L + 4, curY + 10);

    // Spike raster
    if (hasSpikes) {
      const rasY = curY + currentH + gap;
      ctx.fillStyle = PANEL_BG; ctx.fillRect(L, rasY, pw, rasterH);
      ctx.strokeStyle = BORDER; ctx.lineWidth = 1; ctx.strokeRect(L, rasY, pw, rasterH);
      ctx.strokeStyle = "#ff5252"; ctx.lineWidth = 1.5;
      for (const idx of result.spikes) {
        const x = L + ((idx * result.dt - tMin) / (tMax - tMin || 1)) * pw;
        ctx.beginPath(); ctx.moveTo(x, rasY + 2); ctx.lineTo(x, rasY + rasterH - 2); ctx.stroke();
      }
    }

    // X-axis labels
    ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "center";
    const xs = niceStep(tMax - tMin, 6);
    for (let v = Math.ceil(tMin / xs) * xs; v <= tMax; v += xs) {
      const x = L + ((v - tMin) / (tMax - tMin || 1)) * pw;
      ctx.fillText(v.toFixed(0), x, h - 2);
    }
    ctx.textAlign = "right"; ctx.fillText("ms", L + pw, h - 2);
  }, [result, activeTab, fiResult, bifResult, sensResult, precResult, heatmapResult]);

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
      <canvas ref={canvasRef} style={{
        position: "absolute", top: 0, left: 0, width: "100%", height: "100%",
      }} />
    </div>
  );
}
