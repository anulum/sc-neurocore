import { useRef, useEffect, useCallback } from "react";
import { useStudioStore } from "../stores/studio";

const COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373"];
const BG = "#0d1117";
const GRID = "#1a1f2a";
const AXIS = "#484f58";
const BORDER = "#21262d";

function niceStep(range: number, ticks: number): number {
  if (range <= 0) return 1;
  const rough = range / ticks;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const n = rough / mag;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
}

function drawSubplot(
  ctx: CanvasRenderingContext2D,
  x0: number, y0: number, pw: number, ph: number,
  time: number[], datasets: { data: number[]; color: string; label: string }[],
  xMin: number, xMax: number,
  opts?: { spikes?: number[]; dt?: number; isCurrentPlot?: boolean },
) {
  const xRange = xMax - xMin || 1;

  let yMin = Infinity, yMax = -Infinity;
  for (const ds of datasets) {
    for (const v of ds.data) {
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    }
  }
  const yRange = yMax - yMin || 1;
  yMin -= yRange * 0.06;
  yMax += yRange * 0.06;

  const toX = (t: number) => x0 + ((t - xMin) / xRange) * pw;
  const toY = (v: number) => y0 + ph - ((v - yMin) / (yMax - yMin)) * ph;

  // Background
  ctx.fillStyle = "#0a0e14";
  ctx.fillRect(x0, y0, pw, ph);

  // Grid
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 0.5;
  const yStep = niceStep(yMax - yMin, 4);
  const yStart = Math.ceil(yMin / yStep) * yStep;
  ctx.font = "10px monospace";
  ctx.fillStyle = AXIS;
  ctx.textAlign = "right";
  for (let v = yStart; v <= yMax; v += yStep) {
    const y = Math.round(toY(v)) + 0.5;
    if (y < y0 || y > y0 + ph) continue;
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x0 + pw, y);
    ctx.stroke();
    const label = Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3);
    ctx.fillText(label, x0 - 5, y + 3);
  }

  // Border
  ctx.strokeStyle = BORDER;
  ctx.lineWidth = 1;
  ctx.strokeRect(x0, y0, pw, ph);

  // Traces
  for (const ds of datasets) {
    ctx.strokeStyle = ds.color;
    ctx.lineWidth = opts?.isCurrentPlot ? 1.5 : 1.2;
    ctx.beginPath();
    for (let i = 0; i < ds.data.length; i++) {
      const x = toX(time[i]);
      const y = toY(ds.data[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Legend
  ctx.font = "10px monospace";
  datasets.forEach((ds, i) => {
    const lx = x0 + 6 + i * 52;
    ctx.fillStyle = ds.color;
    ctx.fillRect(lx, y0 + 4, 8, 2);
    ctx.textAlign = "left";
    ctx.fillText(ds.label, lx + 11, y0 + 9);
  });

  // Spike markers on voltage plot
  if (opts?.spikes && opts.dt) {
    ctx.strokeStyle = "rgba(255, 82, 82, 0.3)";
    ctx.lineWidth = 1;
    for (const idx of opts.spikes) {
      const x = toX(idx * opts.dt);
      ctx.beginPath();
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y0 + ph);
      ctx.stroke();
    }
  }
}

export default function SimulationPlot() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { result, activeTab, fiResult } = useStudioStore();

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(rect.width) - 4;
    const h = Math.floor(rect.height) - 4;
    if (w < 100 || h < 100) return;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);

    const padL = 56, padR = 12, padT = 8, padB = 22;
    const plotW = w - padL - padR;

    if (activeTab === "fi-curve" && fiResult) {
      const plotH = h - padT - padB;
      drawSubplot(
        ctx, padL, padT, plotW, plotH,
        fiResult.currents, // x axis = currents
        [{ data: fiResult.rates, color: "#4fc3f7", label: "f (Hz)" }],
        fiResult.currents[0],
        fiResult.currents[fiResult.currents.length - 1],
        { isCurrentPlot: true },
      );
      // X axis label
      ctx.fillStyle = AXIS;
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      const xStep = niceStep(
        fiResult.currents[fiResult.currents.length - 1] - fiResult.currents[0], 6
      );
      const xMin = fiResult.currents[0];
      const xMax = fiResult.currents[fiResult.currents.length - 1];
      for (let v = Math.ceil(xMin / xStep) * xStep; v <= xMax; v += xStep) {
        const x = padL + ((v - xMin) / (xMax - xMin || 1)) * plotW;
        ctx.fillText(v.toFixed(0), x, h - 4);
      }
      ctx.textAlign = "right";
      ctx.fillText("I (nA)", padL + plotW, h - 4);
      return;
    }

    if (!result) {
      ctx.fillStyle = AXIS;
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Select a template and simulate", w / 2, h / 2);
      return;
    }

    const time = result.time;
    const vars = Object.keys(result.states);
    const hasSpikes = result.spikes.length > 0;
    const tMin = time[0];
    const tMax = time[time.length - 1];

    // Layout: voltage 60%, current 15%, raster 10%, x-axis labels 15%
    const gapY = 6;
    const rasterH = hasSpikes ? 28 : 0;
    const currentH = 50;
    const usableH = h - padT - padB - (hasSpikes ? gapY : 0) - gapY;
    const voltageH = usableH - currentH - rasterH;

    if (voltageH < 40) return;

    // Voltage subplot
    const voltY0 = padT;
    const voltDs = vars.map((v, i) => ({
      data: result.states[v], color: COLORS[i % COLORS.length], label: v,
    }));
    drawSubplot(ctx, padL, voltY0, plotW, voltageH, time, voltDs, tMin, tMax, {
      spikes: result.spikes, dt: result.dt,
    });

    // Current subplot
    const curY0 = voltY0 + voltageH + gapY;
    drawSubplot(ctx, padL, curY0, plotW, currentH, time, [
      { data: result.current_trace, color: "#ffb74d", label: "I" },
    ], tMin, tMax, { isCurrentPlot: true });

    // Spike raster
    if (hasSpikes) {
      const rasY0 = curY0 + currentH + gapY;
      ctx.fillStyle = "#0a0e14";
      ctx.fillRect(padL, rasY0, plotW, rasterH);
      ctx.strokeStyle = BORDER;
      ctx.lineWidth = 1;
      ctx.strokeRect(padL, rasY0, plotW, rasterH);

      ctx.strokeStyle = "#ff5252";
      ctx.lineWidth = 1.5;
      const toX = (t: number) => padL + ((t - tMin) / (tMax - tMin || 1)) * plotW;
      for (const idx of result.spikes) {
        const x = toX(idx * result.dt);
        ctx.beginPath();
        ctx.moveTo(x, rasY0 + 3);
        ctx.lineTo(x, rasY0 + rasterH - 3);
        ctx.stroke();
      }

      ctx.fillStyle = "#ff5252";
      ctx.font = "9px monospace";
      ctx.textAlign = "left";
      ctx.fillText("spikes", padL + 4, rasY0 + 10);
    }

    // X-axis ticks
    const xStep = niceStep(tMax - tMin, 6);
    ctx.fillStyle = AXIS;
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    for (let v = Math.ceil(tMin / xStep) * xStep; v <= tMax; v += xStep) {
      const x = padL + ((v - tMin) / (tMax - tMin || 1)) * plotW;
      ctx.fillText(v.toFixed(0), x, h - 4);
    }
    ctx.textAlign = "right";
    ctx.fillText("ms", padL + plotW, h - 4);
  }, [result, activeTab, fiResult]);

  useEffect(() => {
    draw();
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [draw]);

  return (
    <div
      ref={containerRef}
      style={{ flex: 1, padding: 8, display: "flex", alignItems: "stretch" }}
    >
      <canvas ref={canvasRef} style={{ borderRadius: 4 }} />
    </div>
  );
}
