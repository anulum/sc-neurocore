import { useEffect, useRef } from "react";
import type {
  TrainingWeightAttachResult,
  TrainingWeightLiveAttachResult,
  TrainingWeightRestorePlan,
  TrainingWeightRestoreResult,
} from "../api/client";
import type { TrainingWeightRestoreVerification } from "../trainingRestore";
import { useStudioStore } from "../stores/studio";
import { buildTrainingEvidenceModel, type TrainingEvidenceModel } from "../trainingEvidence";
import EvidenceSummaryStrip from "./EvidenceSummaryStrip";

function MetricChart({ data, xKey, yKeys, colors, height, yLabel }: {
  data: Record<string, unknown>[];
  xKey: string;
  yKeys: string[];
  colors: string[];
  height: number;
  yLabel: string;
}) {
  if (data.length === 0) return null;
  const allYVals = yKeys.flatMap((k) => data.map((d) => d[k] as number).filter((v) => v != null));
  const yMin = Math.min(...allYVals);
  const yMax = Math.max(...allYVals);
  const yRange = yMax - yMin || 1;
  const xVals = data.map((d) => d[xKey] as number);
  const xMin = Math.min(...xVals);
  const xMax = Math.max(...xVals);
  const xRange = xMax - xMin || 1;

  const w = 320;
  const pad = { top: 8, right: 8, bottom: 20, left: 40 };
  const pw = w - pad.left - pad.right;
  const ph = height - pad.top - pad.bottom;

  const toX = (v: number) => pad.left + ((v - xMin) / xRange) * pw;
  const toY = (v: number) => pad.top + (1 - (v - yMin) / yRange) * ph;

  return (
    <svg width={w} height={height} style={{ display: "block" }}>
      {/* Axes */}
      <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + ph} stroke="var(--border)" strokeWidth={1} />
      <line x1={pad.left} y1={pad.top + ph} x2={pad.left + pw} y2={pad.top + ph} stroke="var(--border)" strokeWidth={1} />
      <text x={2} y={pad.top + ph / 2} fill="var(--text-muted)" fontSize={8} textAnchor="start" transform={`rotate(-90, 8, ${pad.top + ph / 2})`}>{yLabel}</text>
      <text x={pad.left + pw / 2} y={height - 2} fill="var(--text-muted)" fontSize={8} textAnchor="middle">epoch</text>
      {/* Y ticks */}
      {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
        const val = yMin + frac * yRange;
        const y = toY(val);
        return (
          <g key={frac}>
            <line x1={pad.left - 3} y1={y} x2={pad.left} y2={y} stroke="var(--border)" />
            <text x={pad.left - 5} y={y + 3} fill="var(--text-muted)" fontSize={7} textAnchor="end">
              {val < 1 ? val.toFixed(3) : val.toFixed(1)}
            </text>
          </g>
        );
      })}
      {/* Lines */}
      {yKeys.map((key, ki) => {
        const pts = data
          .map((d) => ({ x: d[xKey] as number, y: d[key] as number }))
          .filter((p) => p.y != null);
        if (pts.length < 2) return null;
        const path = pts.map((p, i) => `${i === 0 ? "M" : "L"}${toX(p.x).toFixed(1)},${toY(p.y).toFixed(1)}`).join(" ");
        return <path key={key} d={path} fill="none" stroke={colors[ki]} strokeWidth={1.5} />;
      })}
      {/* Legend */}
      {yKeys.map((key, ki) => (
        <g key={key}>
          <line x1={pad.left + ki * 80} y1={2} x2={pad.left + ki * 80 + 12} y2={2} stroke={colors[ki]} strokeWidth={2} />
          <text x={pad.left + ki * 80 + 15} y={6} fill="var(--text-secondary)" fontSize={8}>{key.replace(/_/g, " ")}</text>
        </g>
      ))}
    </svg>
  );
}

function LayerRateBar({ name, rate }: { name: string; rate: number }) {
  const pct = Math.min(rate * 100, 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
      <span style={{ fontSize: 9, color: "var(--text-muted)", width: 60, overflow: "hidden", textOverflow: "ellipsis" }}>{name}</span>
      <div style={{ flex: 1, height: 6, background: "var(--bg-tertiary)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: "#80cbc4", borderRadius: 3, transition: "width 0.3s" }} />
      </div>
      <span style={{ fontSize: 9, fontFamily: "var(--font-mono)", color: "var(--text-muted)", width: 36, textAlign: "right" }}>
        {(rate * 100).toFixed(1)}%
      </span>
    </div>
  );
}

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
          border: "1px solid var(--border)",
          color: canExport ? "var(--text-secondary)" : "var(--text-muted)",
          cursor: canExport ? "pointer" : "not-allowed",
          fontSize: 10,
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
          border: "1px solid var(--border)",
          color: "var(--text-secondary)",
          cursor: "pointer",
          fontSize: 10,
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
      {(onVerify || onExportVerification) && (
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
                border: "1px solid var(--border)",
                color: "var(--text-secondary)",
                cursor: "pointer",
                fontSize: 10,
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
                border: "1px solid var(--border)",
                color: verification ? "var(--text-secondary)" : "var(--text-muted)",
                cursor: verification ? "pointer" : "not-allowed",
                fontSize: 10,
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
          { label: "Attach", value: "warm_start" },
          { label: "Job", value: attach.job_id },
          { label: "Source", value: attach.source_job_id },
          { label: "Status", value: attach.status },
          { label: "Fingerprint", value: attach.architecture_fingerprint.slice(0, 12) },
        ]}
      />
    </div>
  );
}

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

export default function TrainingMonitor() {
  const {
    trainingStatus, trainingEpochs, trainingSurrogates, trainingConfig,
    trainingJobId, trainingWeightRestorePlan, trainingWeightRestoreVerification,
    trainingWeightMaterialization, trainingWeightAttach, trainingWeightLiveAttach,
    startTraining, stopTraining, setTrainingConfig, loadSurrogates, isSimulating,
    exportTrainingCheckpoint, importTrainingCheckpointText,
    exportTrainingWeightRestoreVerification, verifyTrainingWeightRestoreArtifact,
    materializeTrainingWeights, attachTrainingWeights, liveAttachTrainingWeights,
  } = useStudioStore();

  useEffect(() => { loadSurrogates(); }, [loadSurrogates]);

  const latestEpoch = trainingEpochs.length > 0 ? trainingEpochs[trainingEpochs.length - 1] : null;
  const isRunning = trainingStatus === "running" || trainingStatus === "starting";
  const evidence = buildTrainingEvidenceModel(
    trainingJobId,
    trainingStatus,
    trainingConfig,
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
        <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)" }}>
          Training Monitor
        </span>
        <span style={{
          fontSize: 9, padding: "1px 6px", borderRadius: 3,
          background: isRunning ? "rgba(129, 199, 132, 0.2)" :
                     trainingStatus === "completed" ? "rgba(79, 195, 247, 0.2)" :
                     trainingStatus === "failed" ? "rgba(255, 82, 82, 0.2)" : "var(--bg-tertiary)",
          color: isRunning ? "#81c784" :
                 trainingStatus === "completed" ? "#4fc3f7" :
                 trainingStatus === "failed" ? "#ff5252" : "var(--text-muted)",
        }}>
          {trainingStatus}
        </span>
        {!isRunning && (
          <button
            onClick={startTraining}
            disabled={isSimulating}
            style={{
              background: "#81c784", color: "#0d1117", border: "none",
              padding: "3px 10px", fontSize: 10, cursor: "pointer",
            }}
          >
            Train
          </button>
        )}
        {isRunning && (
          <button
            onClick={stopTraining}
            style={{
              background: "#ff5252", color: "#fff", border: "none",
              padding: "3px 10px", fontSize: 10, cursor: "pointer",
            }}
          >
            Stop
          </button>
        )}
        <TrainingCheckpointControls
          canExport={trainingJobId !== null}
          onExport={() => { void exportTrainingCheckpoint(); }}
          onImportText={(checkpointJson) => { void importTrainingCheckpointText(checkpointJson); }}
        />
        <button
          onClick={() => { void materializeTrainingWeights(); }}
          disabled={trainingJobId === null}
          title="Materialize and verify training weights into confined evidence"
          style={{
            background: "var(--bg-tertiary)",
            border: "1px solid var(--border)",
            color: trainingJobId !== null ? "var(--text-secondary)" : "var(--text-muted)",
            cursor: trainingJobId !== null ? "pointer" : "not-allowed",
            fontSize: 10,
            padding: "3px 8px",
          }}
        >
          Materialize weights
        </button>
        <button
          onClick={() => { void attachTrainingWeights(); }}
          disabled={trainingJobId === null || isRunning}
          title="Warm-start a new training job from the verified weights"
          style={{
            background: "var(--bg-tertiary)",
            border: "1px solid var(--border)",
            color: trainingJobId !== null && !isRunning ? "var(--text-secondary)" : "var(--text-muted)",
            cursor: trainingJobId !== null && !isRunning ? "pointer" : "not-allowed",
            fontSize: 10,
            padding: "3px 8px",
          }}
        >
          Attach (warm-start)
        </button>
        <button
          onClick={() => { void liveAttachTrainingWeights(); }}
          disabled={!isRunning || trainingWeightMaterialization === null}
          title="Attach the verified weights into the running job at the next epoch boundary"
          style={{
            background: "var(--bg-tertiary)",
            border: "1px solid var(--border)",
            color: isRunning && trainingWeightMaterialization !== null ? "var(--text-secondary)" : "var(--text-muted)",
            cursor: isRunning && trainingWeightMaterialization !== null ? "pointer" : "not-allowed",
            fontSize: 10,
            padding: "3px 8px",
          }}
        >
          Live attach
        </button>
      </div>

      <TrainingEvidenceStrip evidence={evidence} />
      <TrainingWeightRestorePlanStrip
        onExportVerification={exportTrainingWeightRestoreVerification}
        onVerify={() => { void verifyTrainingWeightRestoreArtifact(); }}
        restorePlan={trainingWeightRestorePlan}
        verification={trainingWeightRestoreVerification}
      />
      <TrainingWeightMaterializationStrip materialization={trainingWeightMaterialization} />
      <TrainingWeightAttachStrip attach={trainingWeightAttach} />
      <TrainingWeightLiveAttachStrip liveAttach={trainingWeightLiveAttach} />

      {/* Config panel */}
      {!isRunning && (
        <div style={{
          padding: "8px 12px", borderBottom: "1px solid var(--border)",
          display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 6,
          fontSize: 10,
        }}>
          <label style={{ color: "var(--text-secondary)" }}>
            Dataset
            <select value={trainingConfig.dataset} onChange={(e) => setTrainingConfig("dataset", e.target.value)}
              style={{ display: "block", width: "100%", fontSize: 10 }}>
              <option value="synthetic">Synthetic (64D, fast)</option>
              <option value="mnist">MNIST (784D)</option>
            </select>
          </label>
          <label style={{ color: "var(--text-secondary)" }}>
            Epochs
            <input type="number" value={trainingConfig.epochs} min={1} max={100}
              onChange={(e) => setTrainingConfig("epochs", parseInt(e.target.value) || 10)}
              style={{ display: "block", width: "100%", fontSize: 10 }} />
          </label>
          <label style={{ color: "var(--text-secondary)" }}>
            Batch Size
            <input type="number" value={trainingConfig.batch_size} min={8} max={512} step={8}
              onChange={(e) => setTrainingConfig("batch_size", parseInt(e.target.value) || 64)}
              style={{ display: "block", width: "100%", fontSize: 10 }} />
          </label>
          <label style={{ color: "var(--text-secondary)" }}>
            Learning Rate
            <input type="number" value={trainingConfig.lr} min={0.0001} max={0.1} step={0.0001}
              onChange={(e) => setTrainingConfig("lr", parseFloat(e.target.value) || 0.001)}
              style={{ display: "block", width: "100%", fontSize: 10 }} />
          </label>
          <label style={{ color: "var(--text-secondary)" }}>
            Timesteps
            <input type="number" value={trainingConfig.timesteps} min={5} max={100}
              onChange={(e) => setTrainingConfig("timesteps", parseInt(e.target.value) || 25)}
              style={{ display: "block", width: "100%", fontSize: 10 }} />
          </label>
          <label style={{ color: "var(--text-secondary)" }}>
            Surrogate
            <select value={trainingConfig.surrogate} onChange={(e) => setTrainingConfig("surrogate", e.target.value)}
              style={{ display: "block", width: "100%", fontSize: 10 }}>
              {(trainingSurrogates.length > 0 ? trainingSurrogates : [
                { name: "atan_surrogate" }, { name: "fast_sigmoid" }, { name: "superspike" },
                { name: "sigmoid_surrogate" }, { name: "straight_through" }, { name: "triangular" },
              ]).map((s) => (
                <option key={s.name} value={s.name}>{s.name.replace(/_/g, " ")}</option>
              ))}
            </select>
          </label>
          <label style={{ color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 4 }}>
            <input type="checkbox" checked={trainingConfig.learn_beta}
              onChange={(e) => setTrainingConfig("learn_beta", e.target.checked)} />
            Learn beta
          </label>
          <label style={{ color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 4 }}>
            <input type="checkbox" checked={trainingConfig.learn_threshold}
              onChange={(e) => setTrainingConfig("learn_threshold", e.target.checked)} />
            Learn threshold
          </label>
        </div>
      )}

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
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 6 }}>
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
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 6 }}>
                  Learnable Parameters (epoch {latestEpoch.epoch})
                </div>
                <div style={{
                  display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 4,
                  fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--text-muted)",
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
                borderRadius: 4, fontSize: 10, fontFamily: "var(--font-mono)",
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
        {trainingEpochs.length === 0 && !isRunning && (
          <div style={{
            flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
            color: "var(--text-muted)", fontSize: 11, minHeight: 100,
          }}>
            Configure training parameters above, then click Train
          </div>
        )}

        {/* Running indicator */}
        {isRunning && trainingEpochs.length === 0 && (
          <div style={{
            flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
            color: "var(--text-muted)", fontSize: 11, minHeight: 100,
          }}>
            Starting training...
          </div>
        )}
      </div>
    </div>
  );
}
