// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { selectStudioHeatmapPoint } from "../studioHeatmapSelection";
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
import { displayPositionAtTime, rawStepAtTime } from "../simulationRaw";
import { PLOT_AXIS as AXIS } from "../simulationPlotCanvas";
import {
  drawBifurcationView,
  drawFICurveView,
  drawFrequencyResponseView,
  drawHeatmapView,
  drawIsiHistogramView,
  drawSensitivityView,
  drawSpikeTriggeredAverageView,
} from "../plots/analysisViews";
import {
  drawCompareView,
  drawMultiModelView,
  drawPrecisionView,
} from "../plots/comparisonViews";
import { preparePlotCanvas } from "../plots/plotFrame";
import {
  drawCharacterizeView,
  drawNetworkView,
  drawPhasePortraitView,
} from "../plots/stateViews";
import { drawTraceView } from "../plots/traceView";
import { panelTitle } from "../capabilityShell";
import { formatReading, plotDescription, traceDataRows } from "../plotAccessibility";
import EvidenceSummaryStrip from "./EvidenceSummaryStrip";

/** Whichever analysis result the active tab is showing, if any. */
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

/**
 * Read an analysis result's metadata, whichever kind it is.
 *
 * @param result - The active tab's result, or `null`.
 * @returns Its metadata, or `null` when there is no result.
 */
function resultMetadata(result: AnalysisResult): AnalysisResultMetadata | null {
  return result?.analysis_metadata ?? null;
}

/**
 * The plot panel: one canvas, and whichever view the active tab calls for.
 *
 * The drawing lives in `../plots`; this owns the canvas, the interaction state
 * (zoom, drag, crosshair, tooltip) and the choice of view. Splitting it that
 * way is what let the views acquire cases: they are functions taking a context
 * and data, where they used to be branches reachable only through a mounted
 * component and a populated store.
 *
 * @returns The panel.
 */
export default function SimulationPlot() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef({ xMin: NaN, xMax: NaN, yMin: NaN, yMax: NaN });
  const dragRef = useRef<{ startX: number; startY: number; origXMin: number; origXMax: number; origYMin: number; origYMax: number } | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [traceShown, setTraceShown] = useState(false);
  const [dataTableShown, setDataTableShown] = useState(false);
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

  /**
   * On the heatmap, adopt the parameters under the pointer and re-run.
   *
   * @param e - The click.
   */
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
      void selectStudioHeatmapPoint(heatmapResult, xi, yi,
        useStudioStore.getState, useStudioStore.setState);
    }
  }

  /**
   * Zoom the trace's time axis about the pointer.
   *
   * @param e - The wheel event.
   */
  function handleWheel(e: React.WheelEvent) {
    if (activeTab !== "trace" || !result) return;
    e.preventDefault();
    const z = zoomRef.current;
    const time = result.time;
    if (isNaN(z.xMin)) { z.xMin = time[0] ?? 0; z.xMax = time[time.length - 1] ?? 0; }
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

  /**
   * Begin a pan, but only when the view is already zoomed.
   *
   * @param e - The press.
   */
  function handleMouseDown(e: React.MouseEvent) {
    if (activeTab !== "trace" || !result) return;
    const z = zoomRef.current;
    if (isNaN(z.xMin)) return;
    dragRef.current = { startX: e.clientX, startY: e.clientY, origXMin: z.xMin, origXMax: z.xMax, origYMin: z.yMin, origYMax: z.yMax };
  }

  /**
   * Pan a zoomed trace, or move the crosshair and its tooltip.
   *
   * @param e - The movement.
   */
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
      const t0 = isNaN(z.xMin) ? result.time[0] ?? 0 : z.xMin;
      const t1 = isNaN(z.xMax) ? result.time[result.time.length - 1] ?? 0 : z.xMax;
      const tAt = t0 + fracX * (t1 - t0);
      // Display arrays are a projection of the raw steps: read them at the
      // display position nearest to the cursor time and report the raw step.
      const position = displayPositionAtTime(result, tAt);
      const step = rawStepAtTime(result, tAt);
      const vars = Object.keys(result.states);
      const vals = vars.map((v) => {
        const arr = result.states[v] ?? [];
        const i = Math.min(Math.max(position, 0), arr.length - 1);
        return `${v}=${(arr[i] ?? 0).toFixed(2)}`;
      }).join(" ");
      crosshairRef.current = e.clientX - rect.left;
      setTooltip({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        text: `t=${(result.time[position] ?? 0).toFixed(1)} step=${step} ${vals}`,
      });
      draw();
    } else {
      crosshairRef.current = null;
      setTooltip(null);
    }
  }

  /** End a pan. */
  function handleMouseUp() { dragRef.current = null; }
  /** End a pan and clear the crosshair when the pointer leaves the canvas. */
  function handleMouseLeave() { dragRef.current = null; crosshairRef.current = null; setTooltip(null); draw(); }

  /** Return the trace to the whole run. */
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
    const prepared = preparePlotCanvas(
      canvas,
      Math.floor(rect.width),
      Math.floor(rect.height),
      window.devicePixelRatio || 1,
    );
    if (prepared === null) return;
    const { ctx, frame } = prepared;
    // Which view is drawn decides what the canvas says to a screen reader.
    const drewTrace = ((): boolean => {

    // Each view is a function in `../plots`; this chooses one. The conditions
    // are the originals, including which of them fall through to the trace
    // view rather than leaving a blank canvas: a phase portrait of a
    // one-variable system, an ISI view of a run with no histogram, and a
    // spike-triggered average with nothing in it all show the trace instead.
    if (activeTab === "fi-curve" && fiResult) {
      drawFICurveView(ctx, frame, fiResult);
      return false;
    }

    if (!result) {
      ctx.fillStyle = AXIS;
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Select a model and adjust parameters", frame.width / 2, frame.height / 2);
      return false;
    }

    if (activeTab === "phase" && Object.keys(result.states).length >= 2) {
      drawPhasePortraitView(ctx, frame, result, nullclineResult);
      return false;
    }
    if (activeTab === "isi" && result.stats.isi_histogram) {
      drawIsiHistogramView(ctx, frame, result);
      return false;
    }
    if (activeTab === "bifurcation" && bifResult) {
      drawBifurcationView(ctx, frame, bifResult);
      return false;
    }
    if (activeTab === "heatmap" && heatmapResult) {
      drawHeatmapView(ctx, frame, heatmapResult);
      return false;
    }
    if (activeTab === "sensitivity" && sensResult) {
      drawSensitivityView(ctx, frame, sensResult);
      return false;
    }
    if (activeTab === "precision" && precResult) {
      drawPrecisionView(ctx, frame, precResult);
      return false;
    }
    if (activeTab === "compare" && compareResult) {
      drawCompareView(ctx, frame, compareResult);
      return false;
    }
    if (activeTab === "freq" && freqResult) {
      drawFrequencyResponseView(ctx, frame, freqResult);
      return false;
    }
    if (activeTab === "sta" && staResult && staResult.time_ms.length > 0) {
      drawSpikeTriggeredAverageView(ctx, frame, staResult);
      return false;
    }
    if (activeTab === "characterize" && charResult) {
      drawCharacterizeView(ctx, frame, charResult);
      return false;
    }
    if (activeTab === "multi" && multiResults && multiResults.length > 0) {
      drawMultiModelView(ctx, frame, multiResults);
      return false;
    }
    if (activeTab === "network" && networkResult) {
      drawNetworkView(ctx, frame, networkResult);
      return false;
    }

    drawTraceView(ctx, frame, result, {
      crosshair: crosshairRef.current,
      importedTrace,
      zoom: zoomRef.current,
    });
    return true;
    })();
    setTraceShown((shown) => (shown === drewTrace ? shown : drewTrace));
  }, [result, activeTab, fiResult, bifResult, sensResult, precResult, heatmapResult, compareResult, nullclineResult, freqResult, staResult, charResult, multiResults, importedTrace, networkResult]);

  useEffect(() => {
    draw();
    const onResize = () => {
      draw();
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
    };
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
      <button
        type="button"
        aria-pressed={dataTableShown}
        onClick={() => { setDataTableShown((shown) => !shown); }}
        style={{
          position: "absolute", right: 8, bottom: 8, zIndex: 2, fontSize: 10,
          background: "var(--bg-secondary)", color: "var(--text-secondary)",
          border: "1px solid var(--control-border)", borderRadius: 3, padding: "2px 8px", cursor: "pointer",
        }}
      >Data table</button>
      {dataTableShown && (
        <div style={{
          position: "absolute", right: 8, bottom: 36, zIndex: 2, maxHeight: "60%", overflow: "auto",
          background: "var(--bg-secondary)", border: "1px solid var(--border)", padding: 6, fontSize: 10,
        }}>
          {result === null ? (
            <p style={{ margin: 0 }}>Nothing has run yet.</p>
          ) : (
            <table style={{ borderCollapse: "collapse" }}>
              <caption style={{ captionSide: "top", textAlign: "left" }}>
                Trace data: {result.n_steps} steps of {formatReading(result.dt)} ms, {result.spike_count}{" "}
                {result.spike_count === 1 ? "spike" : "spikes"}
              </caption>
              <thead>
                <tr>
                  {["Variable", "Samples shown", "Minimum", "Maximum", "Final"].map((column) => (
                    <th key={column} scope="col" style={{ textAlign: "left", padding: "1px 6px" }}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {traceDataRows(result).map((row) => (
                  <tr key={row.variable}>
                    <th scope="row" style={{ textAlign: "left", padding: "1px 6px" }}>{row.variable}</th>
                    <td style={{ padding: "1px 6px" }}>{row.samples}</td>
                    <td style={{ padding: "1px 6px" }}>{formatReading(row.minimum)}</td>
                    <td style={{ padding: "1px 6px" }}>{formatReading(row.maximum)}</td>
                    <td style={{ padding: "1px 6px" }}>{formatReading(row.final)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
      <canvas ref={canvasRef}
        role="img"
        aria-label={plotDescription(panelTitle(activeTab), result, traceShown)}
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
