import { useRef, useEffect } from "react";
import { useStudioStore } from "../stores/studio";

const PADDING = { top: 20, right: 20, bottom: 30, left: 60 };
const SPIKE_HEIGHT = 30;

export default function SimulationPlot() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { result } = useStudioStore();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !result) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const plotW = w - PADDING.left - PADDING.right;
    const spikeAreaH = result.spikes.length > 0 ? SPIKE_HEIGHT + 10 : 0;
    const plotH = h - PADDING.top - PADDING.bottom - spikeAreaH;

    ctx.fillStyle = "#1e1e1e";
    ctx.fillRect(0, 0, w, h);

    const time = result.time;
    const vars = Object.keys(result.states);
    const vData = result.states[vars[0]];
    if (!vData || vData.length === 0) return;

    const tMin = time[0];
    const tMax = time[time.length - 1];
    let vMin = Infinity;
    let vMax = -Infinity;
    for (const val of vData) {
      if (val < vMin) vMin = val;
      if (val > vMax) vMax = val;
    }
    const vRange = vMax - vMin || 1;
    vMin -= vRange * 0.05;
    vMax += vRange * 0.05;

    function toX(t: number) {
      return PADDING.left + ((t - tMin) / (tMax - tMin)) * plotW;
    }
    function toY(v: number) {
      return PADDING.top + plotH - ((v - vMin) / (vMax - vMin)) * plotH;
    }

    // Grid lines
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = PADDING.top + (plotH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(PADDING.left, y);
      ctx.lineTo(PADDING.left + plotW, y);
      ctx.stroke();
    }

    // Voltage trace
    const colors = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373"];
    vars.forEach((varName, vi) => {
      const data = result.states[varName];
      ctx.strokeStyle = colors[vi % colors.length];
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

    // Spike raster
    if (result.spikes.length > 0) {
      const spikeY = PADDING.top + plotH + 10;
      ctx.strokeStyle = "#ff5252";
      ctx.lineWidth = 1;
      for (const idx of result.spikes) {
        const t = idx * result.dt;
        const x = toX(t);
        ctx.beginPath();
        ctx.moveTo(x, spikeY);
        ctx.lineTo(x, spikeY + SPIKE_HEIGHT);
        ctx.stroke();
      }
    }

    // Axis labels
    ctx.fillStyle = "#aaa";
    ctx.font = "11px monospace";
    ctx.textAlign = "right";
    ctx.fillText(vMax.toFixed(1), PADDING.left - 5, PADDING.top + 10);
    ctx.fillText(vMin.toFixed(1), PADDING.left - 5, PADDING.top + plotH);
    ctx.textAlign = "center";
    ctx.fillText(`${tMin.toFixed(0)}`, PADDING.left, h - 5);
    ctx.fillText(`${tMax.toFixed(0)} ms`, PADDING.left + plotW, h - 5);

    // Legend
    ctx.textAlign = "left";
    vars.forEach((varName, vi) => {
      ctx.fillStyle = colors[vi % colors.length];
      ctx.fillText(varName, PADDING.left + 10 + vi * 60, PADDING.top + 14);
    });

    if (result.spikes.length > 0) {
      ctx.fillStyle = "#ff5252";
      ctx.fillText(
        `${result.spike_count} spikes`,
        PADDING.left + 10,
        PADDING.top + plotH + SPIKE_HEIGHT + 18
      );
    }
  }, [result]);

  return (
    <canvas
      ref={canvasRef}
      width={700}
      height={400}
      style={{ background: "#1e1e1e", borderRadius: 4 }}
    />
  );
}
