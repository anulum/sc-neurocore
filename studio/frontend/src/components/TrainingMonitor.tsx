// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useEffect, useRef, useState } from "react";
import TrainingConfiguration from "./TrainingConfiguration";
import TrainingPreregistrationVerdict from "./TrainingPreregistrationVerdict";
import TrainingConversionResult from "./TrainingConversionResult";
import type {
  TrainingJobSummary,
  TrainingWeightAttachResult,
  TrainingWeightLiveAttachResult,
  TrainingWeightRestorePlan,
  TrainingWeightRestoreResult,
} from "../api/client";
import type { TrainingWeightRestoreVerification } from "../trainingRestore";
import { useStudioStore } from "../stores/studio";
import { buildTrainingEvidenceModel, type TrainingEvidenceModel } from "../trainingEvidence";
import EvidenceSummaryStrip from "./EvidenceSummaryStrip";

/** Text size in the metric charts: the Studio's floor, which their 7–8 px labels were under. */
const CHART_FONT_PX = 11;
/** Advance of one legend character at {@link CHART_FONT_PX}, for placing entries without overlap. */
const CHART_CHAR_PX = 6.6;

/**
 * Say what a metric chart shows, for a reader who cannot see it.
 *
 * @param data - The rows.
 * @param xKey - Which key is the epoch.
 * @param yKeys - The series.
 * @returns One sentence per series: its first and last value and the epochs.
 */
export function metricChartDescription(data: Record<string, unknown>[], xKey: string, yKeys: string[]): string {
  const epochs = data.map((d) => d[xKey] as number);
  const parts = yKeys.map((key) => {
    const values = data.map((d) => d[key] as number).filter((v) => Number.isFinite(v));
    const first = values[0];
    const last = values.at(-1);
    return first === undefined || last === undefined
      ? `${key.replace(/_/g, " ")}: no values`
      : `${key.replace(/_/g, " ")} from ${first.toPrecision(4)} to ${last.toPrecision(4)}`;
  });
  return `${parts.join("; ")}, over epochs ${String(epochs[0] ?? "")}–${String(epochs.at(-1) ?? "")}.`;
}

/**
 * A small multi-series line chart for training metrics.
 *
 * @param props - The rows, which key is the x axis, which are the series, the
 *   colours to draw them in, the height, and the y-axis label.
 * @returns The chart.
 */
function MetricChart({ data, xKey, yKeys, colors, height, yLabel }: {
  data: Record<string, unknown>[];
  xKey: string;
  yKeys: string[];
  colors: string[];
  height: number;
  yLabel: string;
}) {
  if (data.length === 0) return null;
  const allYVals = yKeys.flatMap((k) => data.map((d) => d[k] as number));
  const yMin = Math.min(...allYVals);
  const yMax = Math.max(...allYVals);
  const yRange = yMax - yMin || 1;
  const xVals = data.map((d) => d[xKey] as number);
  const xMin = Math.min(...xVals);
  const xMax = Math.max(...xVals);
  const xRange = xMax - xMin || 1;

  const w = 340;
  const pad = { top: 18, right: 10, bottom: 30, left: 52 };
  const pw = w - pad.left - pad.right;
  const ph = height - pad.top - pad.bottom;

  const toX = (v: number) => pad.left + ((v - xMin) / xRange) * pw;
  const toY = (v: number) => pad.top + (1 - (v - yMin) / yRange) * ph;
  // Whole epochs only, at most six of them.
  const epochStep = Math.max(1, Math.ceil(xRange / 5));
  const epochTicks: number[] = [];
  for (let e = Math.ceil(xMin); e <= xMax; e += epochStep) epochTicks.push(e);
  let legendX = pad.left;
  const legend = yKeys.map((key) => {
    const label = key.replace(/_/g, " ");
    const x = legendX;
    legendX += 18 + label.length * CHART_CHAR_PX + 12;
    return { key, label, x };
  });

  return (
    <svg width={w} height={height} role="img" aria-label={`${yLabel} by epoch: ${metricChartDescription(data, xKey, yKeys)}`}
      style={{ display: "block", fontSize: CHART_FONT_PX }}>
      {/* Axes */}
      <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + ph} stroke="var(--border)" strokeWidth={1} />
      <line x1={pad.left} y1={pad.top + ph} x2={pad.left + pw} y2={pad.top + ph} stroke="var(--border)" strokeWidth={1} />
      <text x={2} y={pad.top + ph / 2} fill="var(--text-muted)" fontSize={CHART_FONT_PX} textAnchor="middle" transform={`rotate(-90, 8, ${pad.top + ph / 2})`}>{yLabel}</text>
      <text x={pad.left + pw / 2} y={height - 2} fill="var(--text-muted)" fontSize={CHART_FONT_PX} textAnchor="middle">epoch</text>
      {/* Y ticks */}
      {[0, 0.5, 1].map((frac) => {
        const val = yMin + frac * yRange;
        const y = toY(val);
        return (
          <g key={frac}>
            <line x1={pad.left - 3} y1={y} x2={pad.left} y2={y} stroke="var(--border)" />
            <text x={pad.left - 5} y={y + 4} fill="var(--text-muted)" fontSize={CHART_FONT_PX} textAnchor="end">
              {val < 1 ? val.toFixed(3) : val.toFixed(1)}
            </text>
          </g>
        );
      })}
      {/* Epoch ticks */}
      {epochTicks.map((e) => (
        <g key={e}>
          <line x1={toX(e)} y1={pad.top + ph} x2={toX(e)} y2={pad.top + ph + 3} stroke="var(--border)" />
          <text x={toX(e)} y={pad.top + ph + 14} fill="var(--text-muted)" fontSize={CHART_FONT_PX} textAnchor="middle">{e}</text>
        </g>
      ))}
      {/* Lines */}
      {yKeys.map((key, ki) => {
        const pts = data.map((d) => ({ x: d[xKey] as number, y: d[key] as number }));
        if (pts.length < 2) return null;
        const path = pts.map((p, i) => `${i === 0 ? "M" : "L"}${toX(p.x).toFixed(1)},${toY(p.y).toFixed(1)}`).join(" ");
        return <path key={key} d={path} fill="none" stroke={colors[ki]} strokeWidth={1.5} />;
      })}
      {/* Legend, each entry placed after the previous one's text */}
      {legend.map(({ key, label, x }, ki) => (
        <g key={key}>
          <line x1={x} y1={7} x2={x + 12} y2={7} stroke={colors[ki]} strokeWidth={2} />
          <text x={x + 16} y={11} fill="var(--text-secondary)" fontSize={CHART_FONT_PX}>{label}</text>
        </g>
      ))}
    </svg>
  );
}

/**
 * One layer's spike rate as a labelled bar.
 *
 * @param props - The layer's name and its rate.
 * @returns The bar.
 */
function LayerRateBar({ name, rate }: { name: string; rate: number }) {
  const pct = Math.min(rate * 100, 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
      <span style={{ fontSize: "var(--fs-meta)", color: "var(--text-muted)", width: 60, overflow: "hidden", textOverflow: "ellipsis" }}>{name}</span>
      <div style={{ flex: 1, height: 6, background: "var(--bg-tertiary)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: "#80cbc4", borderRadius: 3, transition: "width 0.3s" }} />
      </div>
      <span style={{ fontSize: "var(--fs-meta)", fontFamily: "var(--font-mono)", color: "var(--text-muted)", width: 36, textAlign: "right" }}>
        {(rate * 100).toFixed(1)}%
      </span>
    </div>
  );
}

/**
 * What a training run is, stated beside its numbers.
 *
 * @param props - The run's evidence model.
 * @returns The strip.
 */
export function TrainingEvidenceStrip({ evidence }: { evidence: TrainingEvidenceModel }) {
  return (
    <EvidenceSummaryStrip
      variant="banner"
      items={[
        { label: "Evidence", value: evidence.classification },
        { label: "Action", value: evidence.actionKind },
        { label: "Job", value: evidence.jobId },
        { label: "Status", value: evidence.status },
        { label: "Replay", value: evidence.replayRoute },
        { label: "Artifacts", value: `${evidence.statusArtifact} / ${evidence.evidenceArtifact}` },
        { label: "Config", value: evidence.configSummary },
        { label: "Epoch", value: evidence.latestEpoch },
      ]}
    />
  );
}

/**
 * Select a retained run for observation without changing project settings.
 *
 * @param props - Retained jobs, current selection and actions.
 * @returns The run selector and refresh control.
 */
export function TrainingJobPicker({
  jobs,
  selectedJobId,
  loading,
  error,
  onSelect,
  onRefresh,
}: {
  jobs: TrainingJobSummary[];
  selectedJobId: string | null;
  loading: boolean;
  error: string | null;
  onSelect: (jobId: string) => void;
  onRefresh: () => void;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
      <label htmlFor="training-retained-job" style={{ fontSize: "var(--fs-body)" }}>Retained run</label>
      <select
        id="training-retained-job"
        value={jobs.some((job) => job.job_id === selectedJobId) ? selectedJobId ?? "" : ""}
        onChange={(event) => { if (event.target.value) onSelect(event.target.value); }}
        style={{ maxWidth: 220, fontSize: "var(--fs-body)" }}
      >
        <option value="">{jobs.length === 0 ? "No retained runs" : "Choose a run"}</option>
        {jobs.map((job) => (
          <option key={job.job_id} value={job.job_id}>
            {job.job_id} · {job.status}{job.config === null ? " · config not recorded" : ""}
          </option>
        ))}
      </select>
      <button type="button" className="btn-simulate btn btn--ghost" onClick={onRefresh} disabled={loading}>
        {loading ? "Loading…" : "Refresh runs"}
      </button>
      {error !== null && <span role="alert" style={{ color: "#ff5252", fontSize: "var(--fs-body)" }}>{error}</span>}
    </div>
  );
}

/**
 * Export and import controls for a run's checkpoint.
 *
 * @param props - The job and the actions to invoke.
 * @returns The controls.
 */
export function TrainingCheckpointControls({
  canExport,
  onExport,
  onImportText,
}: {
  canExport: boolean;
  onExport: () => void;
  onImportText: (checkpointJson: string) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  return (
    <>
      <button
        onClick={onExport}
        disabled={!canExport}
        title="Export training checkpoint"
        style={{
          background: "var(--bg-tertiary)",
          border: "1px solid var(--control-border)",
          color: canExport ? "var(--text-secondary)" : "var(--text-muted)",
          cursor: canExport ? "pointer" : "not-allowed",
          fontSize: "var(--fs-body)",
          padding: "3px 8px",
        }}
      >
        Export checkpoint
      </button>
      <button
        onClick={() => fileInputRef.current?.click()}
        title="Import training checkpoint"
        style={{
          background: "var(--bg-tertiary)",
          border: "1px solid var(--control-border)",
          color: "var(--text-secondary)",
          cursor: "pointer",
          fontSize: "var(--fs-body)",
          padding: "3px 8px",
        }}
      >
        Import checkpoint
      </button>
      <input
        ref={fileInputRef}
        accept="application/json,.json"
        aria-label="Import training checkpoint file"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (!file) return;
          void file.text().then(onImportText);
          event.target.value = "";
        }}
        style={{ display: "none" }}
        type="file"
      />
    </>
  );
}

/**
 * How a restore intends to fetch and verify a run's weights, before it does.
 *
 * The route and the digest are shown because the restore is only safe if the
 * weights are checked against the configuration they were trained under.
 *
 * @param props - The plan.
 * @returns The strip.
 */
export function TrainingWeightRestorePlanStrip({
  onExportVerification,
  onVerify,
  restorePlan,
  verification,
}: {
  onExportVerification?: () => void;
  onVerify?: () => void;
  restorePlan: TrainingWeightRestorePlan | null;
  verification?: TrainingWeightRestoreVerification | null;
}) {
  if (!restorePlan) return null;

  const weightHash = restorePlan.weights_artifact.sha256.slice(0, 12);
  const metadataHash = restorePlan.metadata_artifact.sha256.slice(0, 12);
  const verifiedHash = verification?.actual_sha256.slice(0, 12) ?? "pending";

  return (
    <div style={{ borderBottom: "1px solid var(--border)" }}>
      <EvidenceSummaryStrip
        variant="banner"
        items={[
          { label: "Schema", value: restorePlan.schema_version },
          { label: "Job", value: restorePlan.source_job_id },
          { label: "Status", value: restorePlan.source_status },
          { label: "Policy", value: restorePlan.loader_policy },
          { label: "Route", value: restorePlan.artifact_route_template },
          { label: "Weights", value: `${restorePlan.weights_artifact.relative_path} #${weightHash}` },
          { label: "Metadata", value: `${restorePlan.metadata_artifact.relative_path} #${metadataHash}` },
          { label: "Verified", value: verifiedHash },
          { label: "Params", value: String(restorePlan.parameter_count) },
        ]}
      />
      {(onVerify ?? onExportVerification) && (
        <div style={{
          background: "var(--bg-primary)",
          display: "flex",
          gap: 6,
          justifyContent: "flex-end",
          padding: "0 12px 6px",
        }}>
          {onVerify && (
            <button
              onClick={onVerify}
              style={{
                background: "var(--bg-tertiary)",
                border: "1px solid var(--control-border)",
                color: "var(--text-secondary)",
                cursor: "pointer",
                fontSize: "var(--fs-body)",
                padding: "3px 8px",
              }}
              title="Verify training weight artifact"
            >
              Verify weights
            </button>
          )}
          {onExportVerification && (
            <button
              disabled={!verification}
              onClick={onExportVerification}
              style={{
                background: "var(--bg-tertiary)",
                border: "1px solid var(--control-border)",
                color: verification ? "var(--text-secondary)" : "var(--text-muted)",
                cursor: verification ? "pointer" : "not-allowed",
                fontSize: "var(--fs-body)",
                padding: "3px 8px",
              }}
              title="Export training weight verification manifest"
            >
              Export verification
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * What a restore actually loaded, as against what it planned to.
 *
 * @param props - The materialisation.
 * @returns The strip.
 */
export function TrainingWeightMaterializationStrip({
  materialization,
}: {
  materialization: TrainingWeightRestoreResult | null;
}) {
  if (!materialization) return null;

  const summary = materialization.materialization;
  return (
    <div style={{ borderBottom: "1px solid var(--border)" }}>
      <EvidenceSummaryStrip
        variant="banner"
        items={[
          { label: "Restore", value: materialization.schema_version },
          { label: "Evidence", value: materialization.evidence_classification },
          { label: "Job", value: materialization.job_id },
          { label: "Source", value: materialization.source_job_id },
          { label: "Status", value: materialization.source_status },
          { label: "Architecture", value: summary.architecture },
          { label: "Params", value: String(summary.parameter_count) },
          { label: "Loaded keys", value: String(summary.loaded_key_count) },
          { label: "Weights", value: summary.weights_sha256.slice(0, 12) },
          { label: "Metadata", value: summary.metadata_sha256.slice(0, 12) },
        ]}
      />
    </div>
  );
}

/**
 * A new run started with another run's weights attached.
 *
 * @param props - The attachment result.
 * @returns The strip.
 */
export function TrainingWeightAttachStrip({
  attach,
}: {
  attach: TrainingWeightAttachResult | null;
}) {
  if (!attach) return null;

  return (
    <div style={{ borderBottom: "1px solid var(--border)" }}>
      <EvidenceSummaryStrip
        variant="banner"
        items={[
          { label: "Attach", value: attach.mode ?? "warm_start" },
          { label: "Job", value: attach.job_id },
          { label: "Source", value: attach.source_job_id },
          { label: "Status", value: attach.status },
          { label: "Fingerprint", value: attach.architecture_fingerprint.slice(0, 12) },
        ]}
      />
    </div>
  );
}

/**
 * Weights attached to a run that was already going.
 *
 * @param props - The attachment result.
 * @returns The strip.
 */
export function TrainingWeightLiveAttachStrip({
  liveAttach,
}: {
  liveAttach: TrainingWeightLiveAttachResult | null;
}) {
  if (!liveAttach) return null;

  return (
    <div style={{ borderBottom: "1px solid var(--border)" }}>
      <EvidenceSummaryStrip
        variant="banner"
        items={[
          { label: "Live attach", value: liveAttach.status },
          { label: "Target", value: liveAttach.target_job_id },
          { label: "Source", value: liveAttach.source_job_id },
          { label: "Fingerprint", value: liveAttach.architecture_fingerprint.slice(0, 12) },
        ]}
      />
    </div>
  );
}

/**
 * The training panel: configuration, live metrics, checkpoints and weights.
 *
 * @returns The panel.
 */
export default function TrainingMonitor() {
  const {
    trainingStatus, trainingEpochs, trainingSurrogates, trainingConfig,
    trainingJobId, trainingWeightRestorePlan, trainingWeightRestoreVerification,
    trainingJobs, trainingJobsLoading, trainingJobsError, trainingObservedConfig,
    trainingPreregistrationVerdict, trainingConversionResult, trainingTargetProfiles, loadTargetProfiles,
    trainingWeightMaterialization, trainingWeightAttach, trainingWeightLiveAttach,
    startTraining, stopTraining, setTrainingConfig, loadSurrogates, isSimulating,
    loadTrainingJobs, selectTrainingJob, authSession,
    exportTrainingCheckpoint, importTrainingCheckpointText,
    exportTrainingWeightRestoreVerification, verifyTrainingWeightRestoreArtifact,
    materializeTrainingWeights, attachTrainingWeights, liveAttachTrainingWeights,
  } = useStudioStore();

  useEffect(() => { void loadSurrogates(); }, [loadSurrogates]);
  useEffect(() => { void loadTargetProfiles(); }, [loadTargetProfiles]);
  useEffect(() => { void loadTrainingJobs(); }, [loadTrainingJobs, authSession]);

  const [trainingInputReady, setTrainingInputReady] = useState(true);
  const latestEpoch = trainingEpochs[trainingEpochs.length - 1] ?? null;
  const isActive = ["running", "starting", "stopping", "unknown", "disconnected"]
    .includes(trainingStatus);
  const isUncertain = trainingStatus === "unknown" || trainingStatus === "disconnected";
  const canStop = trainingJobId !== null
    && (trainingStatus === "running" || trainingStatus === "stopping");
  const canLiveAttach = trainingStatus === "running";
  const evidence = buildTrainingEvidenceModel(
    trainingJobId,
    trainingStatus,
    trainingJobId === null ? trainingConfig : trainingObservedConfig,
    latestEpoch,
  );

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "auto" }}>
      {/* Header */}
      <div style={{
        padding: "8px 12px", background: "var(--bg-secondary)",
        borderBottom: "1px solid var(--border)",
        display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap",
      }}>
        <h2 className="panel-header" style={{ margin: 0 }}>Training monitor</h2>
        <span style={{
          fontSize: "var(--fs-meta)", padding: "1px 6px", borderRadius: 3,
          background: isUncertain ? "rgba(255, 193, 7, 0.2)" :
                     isActive ? "rgba(129, 199, 132, 0.2)" :
                     trainingStatus === "completed" ? "rgba(79, 195, 247, 0.2)" :
                     trainingStatus === "failed" ? "rgba(255, 82, 82, 0.2)" : "var(--bg-tertiary)",
          color: isUncertain ? "#ffc107" :
                 isActive ? "#81c784" :
                 trainingStatus === "completed" ? "#4fc3f7" :
                 trainingStatus === "failed" ? "#ff5252" : "var(--text-muted)",
        }}>
          {trainingStatus}
        </span>
        <TrainingJobPicker
          jobs={trainingJobs}
          selectedJobId={trainingJobId}
          loading={trainingJobsLoading}
          error={trainingJobsError}
          onSelect={(jobId) => { void selectTrainingJob(jobId); }}
          onRefresh={() => { void loadTrainingJobs(); }}
        />
        {!isActive && (
          <button
            onClick={() => { void startTraining(); }}
            disabled={isSimulating || !trainingInputReady}
            style={{
              background: "#81c784", color: "#0d1117", border: "none",
              padding: "3px 10px", fontSize: "var(--fs-body)", cursor: "pointer",
            }}
          >
            Train
          </button>
        )}
        {isActive && (
          <button
            onClick={() => { void stopTraining(); }}
            disabled={!canStop}
            title={canStop ? "Request cooperative stop" : "Refresh to confirm the run before stopping"}
            style={{
              background: "#ff5252", color: "#fff", border: "none",
              padding: "3px 10px", fontSize: "var(--fs-body)", cursor: "pointer",
            }}
          >
            Stop
          </button>
        )}
        <TrainingCheckpointControls
          canExport={trainingJobId !== null && trainingObservedConfig !== null}
          onExport={() => { void exportTrainingCheckpoint(); }}
          onImportText={(checkpointJson) => { void importTrainingCheckpointText(checkpointJson); }}
        />
        <button
          onClick={() => { void materializeTrainingWeights(); }}
          disabled={trainingJobId === null}
          title="Materialize and verify training weights into confined evidence"
          style={{
            background: "var(--bg-tertiary)",
            border: "1px solid var(--control-border)",
            color: trainingJobId !== null ? "var(--text-secondary)" : "var(--text-muted)",
            cursor: trainingJobId !== null ? "pointer" : "not-allowed",
            fontSize: "var(--fs-body)",
            padding: "3px 8px",
          }}
        >
          Materialize weights
        </button>
        <button
          onClick={() => { void attachTrainingWeights(); }}
          disabled={trainingJobId === null || isActive || !trainingInputReady}
          title="Warm-start a new training job from the verified weights"
          style={{
            background: "var(--bg-tertiary)",
            border: "1px solid var(--control-border)",
            color: trainingJobId !== null && !isActive ? "var(--text-secondary)" : "var(--text-muted)",
            cursor: trainingJobId !== null && !isActive ? "pointer" : "not-allowed",
            fontSize: "var(--fs-body)",
            padding: "3px 8px",
          }}
        >
          Attach (warm-start)
        </button>
        <button type="button" className="btn-simulate btn btn--ghost"
          onClick={() => { void attachTrainingWeights("exact_resume"); }}
          disabled={trainingJobId === null || trainingStatus !== "completed" || !trainingInputReady}
          title="Continue the saved optimiser, random state and epoch position; keep the original input unchanged"
        >Resume from checkpoint</button>
        <button
          onClick={() => { void liveAttachTrainingWeights(); }}
          disabled={!canLiveAttach || trainingWeightMaterialization === null}
          title="Attach the verified weights into the running job at the next epoch boundary"
          style={{
            background: "var(--bg-tertiary)",
            border: "1px solid var(--control-border)",
            color: canLiveAttach && trainingWeightMaterialization !== null ? "var(--text-secondary)" : "var(--text-muted)",
            cursor: canLiveAttach && trainingWeightMaterialization !== null ? "pointer" : "not-allowed",
            fontSize: "var(--fs-body)",
            padding: "3px 8px",
          }}
        >
          Live attach
        </button>
      </div>

      <TrainingEvidenceStrip evidence={evidence} />
      <TrainingConversionResult result={trainingConversionResult}
        target={trainingObservedConfig?.target_profile} />
      <TrainingPreregistrationVerdict verdict={trainingPreregistrationVerdict} />
      <TrainingWeightRestorePlanStrip
        onExportVerification={exportTrainingWeightRestoreVerification}
        onVerify={() => { void verifyTrainingWeightRestoreArtifact(); }}
        restorePlan={trainingWeightRestorePlan}
        verification={trainingWeightRestoreVerification}
      />
      <TrainingWeightMaterializationStrip materialization={trainingWeightMaterialization} />
      <TrainingWeightAttachStrip attach={trainingWeightAttach} />
      <TrainingWeightLiveAttachStrip liveAttach={trainingWeightLiveAttach} />

      {!isActive && <TrainingConfiguration config={trainingConfig} surrogates={trainingSurrogates}
        targetProfiles={trainingTargetProfiles}
        setConfig={setTrainingConfig} onReadyChange={setTrainingInputReady} />}

      {/* Charts */}
      <div style={{ padding: 12, flex: 1, overflow: "auto" }}>
        {trainingEpochs.length > 0 && (
          <>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
              <div>
                <MetricChart
                  data={trainingEpochs as unknown as Record<string, unknown>[]}
                  xKey="epoch"
                  yKeys={["train_loss", "val_loss"]}
                  colors={["#4fc3f7", "#ff8a80"]}
                  height={160}
                  yLabel="loss"
                />
              </div>
              <div>
                <MetricChart
                  data={trainingEpochs as unknown as Record<string, unknown>[]}
                  xKey="epoch"
                  yKeys={["train_accuracy", "val_accuracy"]}
                  colors={["#81c784", "#ce93d8"]}
                  height={160}
                  yLabel="accuracy"
                />
              </div>
            </div>

            {/* Layer spike rates */}
            {latestEpoch && Object.keys(latestEpoch.layer_spike_rates).length > 0 && (
              <div style={{ marginTop: 16 }}>
                <div style={{ fontSize: "var(--fs-body)", fontWeight: 600, color: "var(--text-secondary)", marginBottom: 6 }}>
                  Layer Spike Rates (epoch {latestEpoch.epoch})
                </div>
                {Object.entries(latestEpoch.layer_spike_rates).map(([name, rate]) => (
                  <LayerRateBar key={name} name={name} rate={rate} />
                ))}
              </div>
            )}

            {/* Parameter evolution */}
            {latestEpoch && Object.keys(latestEpoch.param_snapshot).length > 0 && (
              <div style={{ marginTop: 16 }}>
                <div style={{ fontSize: "var(--fs-body)", fontWeight: 600, color: "var(--text-secondary)", marginBottom: 6 }}>
                  Learnable Parameters (epoch {latestEpoch.epoch})
                </div>
                <div style={{
                  display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 4,
                  fontSize: "var(--fs-body)", fontFamily: "var(--font-mono)", color: "var(--text-muted)",
                }}>
                  {Object.entries(latestEpoch.param_snapshot).map(([name, val]) => (
                    <div key={name}>{name.split(".").pop()}: {val.toFixed(4)}</div>
                  ))}
                </div>
              </div>
            )}

            {/* Latest numbers */}
            {latestEpoch && (
              <div style={{
                marginTop: 16, padding: 8, background: "var(--bg-secondary)",
                borderRadius: 4, fontSize: "var(--fs-body)", fontFamily: "var(--font-mono)",
                color: "var(--text-secondary)",
                display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 4,
              }}>
                <div>Train Loss: {latestEpoch.train_loss.toFixed(4)}</div>
                <div>Val Loss: {latestEpoch.val_loss.toFixed(4)}</div>
                <div>Train Acc: {(latestEpoch.train_accuracy * 100).toFixed(1)}%</div>
                <div>Val Acc: {(latestEpoch.val_accuracy * 100).toFixed(1)}%</div>
              </div>
            )}
          </>
        )}

        {/* Empty state */}
        {trainingEpochs.length === 0 && !isActive && (
          <div style={{
            flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
            color: "var(--text-muted)", fontSize: "var(--fs-body)", minHeight: 100,
          }}>
            {trainingJobId === null
              ? "Configure training parameters above, then click Train"
              : "No recorded epochs for this retained run"}
          </div>
        )}

        {/* Running indicator */}
        {isActive && trainingEpochs.length === 0 && (
          <div style={{
            flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
            color: "var(--text-muted)", fontSize: "var(--fs-body)", minHeight: 100,
          }}>
            {trainingStatus === "unknown" || trainingStatus === "disconnected"
              ? "Run status is uncertain; refresh runs to reconnect"
              : trainingStatus === "stopping" ? "Stopping training..." : "Starting training..."}
          </div>
        )}
      </div>
    </div>
  );
}
