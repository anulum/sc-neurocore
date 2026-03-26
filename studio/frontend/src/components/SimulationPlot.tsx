import { useRef, useEffect, useCallback } from "react";
import { useStudioStore } from "../stores/studio";

const PAD = { top: 24, right: 16, bottom: 4, left: 56 };
const RASTER_H = 36;
const RASTER_GAP = 8;
const COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373"];
const GRID_COLOR = "#1e2530";
const AXIS_COLOR = "#484f58";
const BG = "#0d1117";

function niceStep(range: number, targetTicks: number): number {
  const rough = range / targetTicks;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return nice * mag;
}

export default function SimulationPlot() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { result } = useStudioStore();

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !result) return;

    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);

    const time = result.time;
    const vars = Object.keys(result.states);
    const hasSpikes = result.spikes.length > 0;
    const rasterArea = hasSpikes ? RASTER_H + RASTER_GAP : 0;
    const plotW = w - PAD.left - PAD.right;
    const plotH = h - PAD.top - PAD.bottom - rasterArea - 24;

    if (plotW < 10 || plotH < 10) return;

    const tMin = time[0];
    const tMax = time[time.length - 1];
    const tRange = tMax - tMin || 1;

    // Compute Y range across all variables
    let vMin = Infinity;
    let vMax = -Infinity;
    for (const v of vars) {
      for (const val of result.states[v]) {
        if (val < vMin) vMin = val;
        if (val > vMax) vMax = val;
      }
    }
    const vRange = vMax - vMin || 1;
    vMin -= vRange * 0.08;
    vMax += vRange * 0.08;

    const toX = (t: number) => PAD.left + ((t - tMin) / tRange) * plotW;
    const toY = (v: number) => PAD.top + plotH - ((v - vMin) / (vMax - vMin)) * plotH;

    // Y grid + tick labels
    const yStep = niceStep(vMax - vMin, 5);
    const yStart = Math.ceil(vMin / yStep) * yStep;
    ctx.strokeStyle = GRID_COLOR;
    ctx.lineWidth = 1;
    ctx.fillStyle = AXIS_COLOR;
    ctx.font = "10px monospace";
    ctx.textAlign = "right";
    for (let yVal = yStart; yVal <= vMax; yVal += yStep) {
      const y = Math.round(toY(yVal)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(PAD.left + plotW, y);
      ctx.stroke();
      ctx.fillText(yVal.toFixed(1), PAD.left - 6, y + 3);
    }

    // X grid + tick labels
    const xStep = niceStep(tRange, 6);
    const xStart = Math.ceil(tMin / xStep) * xStep;
    ctx.textAlign = "center";
    const xLabelY = PAD.top + plotH + 14;
    for (let xVal = xStart; xVal <= tMax; xVal += xStep) {
      const x = Math.round(toX(xVal)) + 0.5;
      ctx.beginPath();
      ctx.strokeStyle = GRID_COLOR;
      ctx.moveTo(x, PAD.top);
      ctx.lineTo(x, PAD.top + plotH);
      ctx.stroke();
      ctx.fillStyle = AXIS_COLOR;
      ctx.fillText(xVal.toFixed(0), x, xLabelY);
    }

    // X axis label
    ctx.fillStyle = AXIS_COLOR;
    ctx.font = "10px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("ms", PAD.left + plotW, xLabelY);

    // Plot border
    ctx.strokeStyle = "#21262d";
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD.left, PAD.top, plotW, plotH);

    // Variable traces
    vars.forEach((varName, vi) => {
      const data = result.states[varName];
      ctx.strokeStyle = COLORS[vi % COLORS.length];
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const x = toX(time[i]);
        const y = toY(data[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    // Legend
    ctx.font = "11px monospace";
    vars.forEach((varName, vi) => {
      const x = PAD.left + 8 + vi * 56;
      ctx.fillStyle = COLORS[vi % COLORS.length];
      ctx.fillRect(x, PAD.top - 16, 10, 3);
      ctx.fillText(varName, x + 14, PAD.top - 11);
    });

    // Spike raster
    if (hasSpikes) {
      const rasterTop = PAD.top + plotH + 24;

      // Raster background
      ctx.fillStyle = "#0f1318";
      ctx.fillRect(PAD.left, rasterTop, plotW, RASTER_H);
      ctx.strokeStyle = "#21262d";
      ctx.strokeRect(PAD.left, rasterTop, plotW, RASTER_H);

      // Spike lines
      ctx.strokeStyle = "#ff5252";
      ctx.lineWidth = 1.5;
      for (const idx of result.spikes) {
        const t = idx * result.dt;
        const x = toX(t);
        ctx.beginPath();
        ctx.moveTo(x, rasterTop + 4);
        ctx.lineTo(x, rasterTop + RASTER_H - 4);
        ctx.stroke();
      }

      // Spike count label
      ctx.fillStyle = "#ff5252";
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      ctx.fillText(
        `${result.spike_count} spikes`,
        PAD.left + plotW - 4,
        rasterTop + RASTER_H - 6
      );
    }
  }, [result]);

  useEffect(() => {
    draw();
    const handleResize = () => draw();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [draw]);

  return (
    <div ref={containerRef} className="plot-container" style={{ flex: 1 }}>
      <canvas ref={canvasRef} style={{ borderRadius: 6 }} />
    </div>
  );
}
