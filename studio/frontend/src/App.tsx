// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useEffect, useState } from "react";
import { useStudioStore } from "./stores/studio";
import type { ViewTab } from "./stores/studio";
import { panelCapabilityState } from "./capabilityShell";
import type { PanelCapabilityState, PanelKey } from "./capabilityShell";
import { downloadBrowserArtefact } from "./browserArtefactDownload";
import {
  analysisCartDraft,
  buildEvidenceCartExport,
  emptyEvidenceCart,
  enqueueEvidenceCartArtefact,
  evidenceCartExportFilename,
  evidenceCartExportToBlob,
  simulationCartDraft,
  type EvidenceCart,
  type EvidenceCartExportBundle,
} from "./evidenceCart";
import EvidenceCartStrip from "./components/EvidenceCartStrip";
import DevelopmentPreviewBanner from "./components/DevelopmentPreviewBanner";
import TemplateLibrary from "./components/TemplateLibrary";
import EquationEditor from "./components/EquationEditor";
import ParameterSliders from "./components/ParameterSliders";
import SimulationPlot from "./components/SimulationPlot";
import ModelInfo from "./components/ModelInfo";
import ModelDocViewer from "./components/ModelDocViewer";
import BackendMatrix from "./components/BackendMatrix";
import DclsPanel from "./components/DclsPanel";
import SpikeStats from "./components/SpikeStats";
import ModelBrowser from "./components/ModelBrowser";
import VerilogPreview from "./components/VerilogPreview";
import MultiModelPicker from "./components/MultiModelPicker";
import ModelComparison from "./components/ModelComparison";
import NetworkControls from "./components/NetworkControls";
import ComparePanel from "./components/ComparePanel";
import KeyboardHelp from "./components/KeyboardHelp";
import StatusBar from "./components/StatusBar";
import CompilerInspector from "./components/CompilerInspector";
import SynthesisDashboard from "./components/SynthesisDashboard";
import TrainingMonitor from "./components/TrainingMonitor";
import NetworkCanvas from "./components/NetworkCanvas";
import OnboardingOverlay from "./components/OnboardingOverlay";
import CapabilityStrip from "./components/CapabilityStrip";
import AdminPanel from "./components/AdminPanel";
import AuthControl from "./components/AuthControl";
import ProjectEvidenceStrip from "./components/ProjectEvidenceStrip";
import GuidedFlowPanel from "./components/GuidedFlowPanel";
import OperatorWorkbenchPanel from "./components/OperatorWorkbenchPanel";
import StudioReadinessPanel from "./components/StudioReadinessPanel";
import { buildProjectEvidenceModel } from "./projectEvidence";
import { computeGuidedFlowState } from "./guidedFlowState";
import type { GuidedFlowCapabilityMap, GuidedFlowInputs } from "./guidedFlowState";
import { buildGuidedRunController } from "./guidedRunController";
import { buildOperatorWorkbenchState } from "./operatorWorkbenchState";
import type { OperatorWorkbenchEvidenceTarget } from "./operatorWorkbenchState";
import { buildStudioReadinessModel } from "./studioReadiness";

function Tab({ active, color, label, onClick, disabled, title }: {
  active: boolean; color: string; label: string; onClick: () => void; disabled?: boolean; title?: string;
}) {
  return (
    <button onClick={onClick} disabled={disabled} title={title} style={{
      padding: "2px 6px", fontSize: 9, fontWeight: 600,
      fontFamily: "var(--font-ui)", lineHeight: 1.4,
      background: active ? color : "transparent",
      color: active ? "var(--bg-primary)" : disabled ? "var(--text-muted)" : "var(--text-secondary)",
      border: "1px solid var(--border)", cursor: disabled ? "not-allowed" : "pointer", whiteSpace: "nowrap",
      opacity: disabled ? 0.45 : 1,
    }}>{label}</button>
  );
}

function Btn({ label, onClick, disabled, color, outline, title }: {
  label: string; onClick: () => void; disabled?: boolean;
  color?: string; outline?: boolean; title?: string;
}) {
  return (
    <button className="btn-simulate" onClick={onClick} disabled={disabled} title={title} style={{
      background: outline ? "transparent" : (color || "var(--accent)"),
      border: outline ? "1px solid var(--border)" : "none",
      color: outline ? "var(--text-muted)" : "var(--bg-primary)",
      padding: "2px 7px", fontSize: 10,
    }}>{label}</button>
  );
}

function CapabilityUnavailable({ state }: { state: PanelCapabilityState }) {
  return (
    <div className="capability-blocked-panel">
      <div className="capability-blocked-title">{state.title}</div>
      <div className="capability-blocked-status">{state.status}</div>
      <p>{state.message}</p>
      {state.requirements.length > 0 && (
        <ul>
          {state.requirements.map((requirement) => <li key={requirement}>{requirement}</li>)}
        </ul>
      )}
      <div className="capability-blocked-meta">
        {state.evidence.length > 0 && <span>Evidence: {state.evidence.join(", ")}</span>}
        {state.docsPath && <a href={`/${state.docsPath}`} target="_blank" rel="noreferrer">Documentation</a>}
      </div>
    </div>
  );
}

export default function App() {
  const [guidedTrainingSkipped, setGuidedTrainingSkipped] = useState(false);
  const [evidenceCart, setEvidenceCart] = useState<EvidenceCart>(() => emptyEvidenceCart());
  const [evidenceCartExport, setEvidenceCartExport] = useState<EvidenceCartExportBundle | null>(
    null,
  );
  const [evidenceCartError, setEvidenceCartError] = useState<string | null>(null);
  const s = useStudioStore();
  const loadCapabilities = s.loadCapabilities;
  const loadAuditStatus = s.loadAuditStatus;
  const loadAuthSession = s.loadAuthSession;
  const loadOperatorStatus = s.loadOperatorStatus;
  const loadPresets = s.loadPresets;
  const vars = s.result ? Object.keys(s.result.states) : [];
  const hasPhase = vars.length >= 2;
  const hasISI = s.result?.stats?.isi_histogram != null;
  const paramKeys = Object.keys(s.sourceMode === "model" ? s.modelParams : s.odeParams);
  const pattern = s.result?.pattern;
  const panelState = (panelKey: PanelKey) => panelCapabilityState(s.capabilities, panelKey);
  const activePanelState = panelState(s.activeTab as PanelKey);
  const panelUnavailable = (panelKey: PanelKey) => !panelState(panelKey).available;

  const guidedFlowInputs: GuidedFlowInputs = {
    modelSelected: s.sourceMode === "ode" ? s.equations.length > 0 : s.selectedModelName.length > 0,
    simulationComplete: s.result !== null,
    analysisComplete: [
      s.fiResult,
      s.bifResult,
      s.sensResult,
      s.precResult,
      s.heatmapResult,
      s.compareResult,
      s.nullclineResult,
      s.freqResult,
      s.charResult,
    ].some((analysis) => analysis !== null),
    trainingComplete: s.trainingEpochs.length > 0,
    trainingSkipped: guidedTrainingSkipped,
    compileComplete: s.compileTraceability !== null,
    synthesisComplete: s.synthResult !== null || s.multiTargetResult !== null,
    evidenceExported: s.evidenceBundle !== null
      || s.projectEvidenceBundle !== null
      || s.compileEvidenceBundle !== null
      || s.synthesisEvidenceBundle !== null
      || evidenceCartExport !== null,
  };
  const guidedFlowCapabilities: GuidedFlowCapabilityMap = {
    design: true,
    simulate: !panelUnavailable("trace"),
    analyse: !panelUnavailable("fi-curve"),
    train: !panelUnavailable("train"),
    compile: !panelUnavailable("verilog"),
    synthesise: !panelUnavailable("synth"),
    export: true,
  };
  const guidedFlow = computeGuidedFlowState(guidedFlowInputs, guidedFlowCapabilities);
  const studioReadiness = buildStudioReadinessModel(s.operatorStatus);
  const operatorWorkbench = buildOperatorWorkbenchState({
    compileBundleExported: s.compileEvidenceBundle !== null,
    compileComplete: s.compileTraceability !== null,
    guidedFlow,
    isSimulating: s.isSimulating,
    modelCount: s.models.length,
    operatorStatus: s.operatorStatus,
    progressMessage: s.progressMsg,
    projectBundleExported: s.projectEvidenceBundle !== null,
    projectName: s.projectSaveResult?.name ?? null,
    savedSessionCount: s.savedSessions.length,
    selectedModelName: s.selectedModelName,
    serverProjectCount: s.serverProjects.length,
    simulationResult: s.result,
    sourceMode: s.sourceMode,
    synthesisBundleExported: s.synthesisEvidenceBundle !== null,
    synthesisComplete: s.synthResult !== null || s.multiTargetResult !== null,
  });
  const panelControl = (panelKey: PanelKey) => {
    const capabilityState = panelState(panelKey);
    return {
      disabled: !capabilityState.available,
      title: capabilityState.message,
    };
  };
  const activatePanel = (panelKey: ViewTab) => {
    const capabilityState = panelState(panelKey as PanelKey);
    if (!capabilityState.available) return;
    s.setActiveTab(panelKey);
  };
  const exportProjectEvidenceBundle = () => {
    if (s.projectSaveResult === null) {
      return;
    }
    void s.createEvidenceBundleForSurface("project", {
      audit_limit: 100,
      analysis_results: [],
      command_replay: {
        method: "POST",
        note: `project ${s.projectSaveResult.name}`,
        request_sha256: s.projectSaveResult.project_sha256,
        route: "/api/project/save",
      },
      default_flow_attestations: [],
      default_flow_runs: [],
      include_audit: true,
      job_ids: [],
      model_scan_results: [],
      project_name: s.projectSaveResult.name,
      simulation_results: [],
      weight_restore_results: [],
      weight_restore_attach_results: [],
    });
  };
  const exportCompileEvidenceBundle = () => {
    if (s.compileTraceability === null) {
      return;
    }
    void s.createEvidenceBundleForSurface("compile", {
      audit_limit: 100,
      analysis_results: [],
      command_replay: {
        method: "POST",
        note: `compile trace ${s.compileTraceability.traceability_sha256}`,
        request_sha256: s.compileTraceability.input_sha256,
        route: "/api/ir/emit-sv-direct",
      },
      default_flow_attestations: [],
      default_flow_runs: [],
      include_audit: true,
      job_ids: [],
      model_scan_results: [],
      project_name: `compile-${s.compileTraceability.output.module_name}`,
      simulation_results: [],
      weight_restore_results: [],
      weight_restore_attach_results: [],
    });
  };
  const exportSynthesisEvidenceBundle = () => {
    const activeJobId = s.multiTargetResult === null
      ? s.latestSynthesisJobId
      : s.latestMultiTargetSynthesisJobId;
    if (activeJobId === null) {
      return;
    }
    void s.createEvidenceBundleForSurface("synthesis", {
      audit_limit: 100,
      analysis_results: [],
      command_replay: null,
      default_flow_attestations: [],
      default_flow_runs: [],
      include_audit: true,
      job_ids: [activeJobId],
      model_scan_results: [],
      project_name: s.multiTargetResult === null
        ? `synthesis-${s.synthResult?.target ?? s.synthTarget}`
        : "synthesis-multi-target",
      simulation_results: [],
      weight_restore_results: [],
      weight_restore_attach_results: [],
    });
  };
  const operateWorkbenchEvidenceBundle = (target: OperatorWorkbenchEvidenceTarget) => {
    switch (target) {
      case "compile":
        if (s.compileEvidenceBundle !== null) {
          activatePanel("verilog");
          return;
        }
        exportCompileEvidenceBundle();
        return;
      case "project":
        if (s.projectEvidenceBundle !== null) {
          activatePanel(s.sourceMode === "model" ? "trace" : "ir");
          return;
        }
        exportProjectEvidenceBundle();
        return;
      case "synthesis":
        if (s.synthesisEvidenceBundle !== null) {
          activatePanel("synth");
          return;
        }
        exportSynthesisEvidenceBundle();
        return;
    }
  };
  const queueSimulationIntoEvidenceCart = () => {
    const result = useStudioStore.getState().result;
    const modelName = useStudioStore.getState().selectedModelName || "ode";
    if (result === null) {
      return;
    }
    setEvidenceCart((cart) => {
      const queued = enqueueEvidenceCartArtefact(
        cart,
        simulationCartDraft(modelName, {
          dt: result.dt,
          model: modelName,
          n_steps: result.n_steps,
          spike_count: result.spike_count,
          spikes: result.spikes,
          time: result.time,
        }),
      );
      if (!queued.ok) {
        setEvidenceCartError(queued.error);
        return cart;
      }
      setEvidenceCartError(null);
      return queued.cart;
    });
  };
  const queueAnalysisIntoEvidenceCart = () => {
    const state = useStudioStore.getState();
    const modelName = state.selectedModelName || "ode";
    const payload = {
      bif: state.bifResult,
      char: state.charResult,
      compare: state.compareResult,
      fi: state.fiResult,
      freq: state.freqResult,
      heatmap: state.heatmapResult,
      model: modelName,
      nullcline: state.nullclineResult,
      prec: state.precResult,
      sens: state.sensResult,
    };
    const hasAnalysis = [
      payload.fi,
      payload.bif,
      payload.sens,
      payload.prec,
      payload.heatmap,
      payload.compare,
      payload.nullcline,
      payload.freq,
      payload.char,
    ].some((entry) => entry !== null);
    if (!hasAnalysis) {
      return;
    }
    setEvidenceCart((cart) => {
      const queued = enqueueEvidenceCartArtefact(
        cart,
        analysisCartDraft(modelName, payload),
      );
      if (!queued.ok) {
        setEvidenceCartError(queued.error);
        return cart;
      }
      setEvidenceCartError(null);
      return queued.cart;
    });
  };
  const exportSessionEvidenceCart = async () => {
    const bundle = await buildEvidenceCartExport(evidenceCart);
    if ("error" in bundle) {
      setEvidenceCartError(bundle.error);
      throw new Error(bundle.error);
    }
    downloadBrowserArtefact(
      evidenceCartExportToBlob(bundle),
      evidenceCartExportFilename(bundle),
    );
    setEvidenceCartExport(bundle);
    setEvidenceCartError(null);
  };
  const guidedRun = buildGuidedRunController({
    capabilityMessages: {
      analyse: panelState("fi-curve").message,
      compile: panelState("verilog").message,
      simulate: panelState("trace").message,
      synthesise: panelState("synth").message,
      train: panelState("train").message,
    },
    exportReady: operatorWorkbench.evidenceActionEnabled || evidenceCart.items.length > 0,
    flow: guidedFlow,
    sourceMode: s.sourceMode,
  }, {
    exportEvidence: async () => {
      if (evidenceCart.items.length > 0) {
        await exportSessionEvidenceCart();
        return;
      }
      if (operatorWorkbench.evidenceExportTarget === null) {
        throw new Error("No evidence export target is available.");
      }
      operateWorkbenchEvidenceBundle(operatorWorkbench.evidenceExportTarget);
    },
    runAnalysis: async () => {
      await s.runFICurve();
      queueAnalysisIntoEvidenceCart();
    },
    runCompile: async () => {
      await s.runEmitSV();
    },
    runSimulation: async () => {
      await s.runSimulation();
      queueSimulationIntoEvidenceCart();
    },
    runSynthesis: async () => {
      await s.runSynthesis();
    },
    skipTraining: async () => {
      setGuidedTrainingSkipped(true);
    },
  });
  const refreshStudioReadiness = () => {
    void loadCapabilities();
    void loadOperatorStatus();
  };

  useEffect(() => {
    void loadCapabilities();
    void loadAuthSession();
    void loadAuditStatus();
    void loadOperatorStatus();
    void loadPresets();
  }, [loadAuditStatus, loadAuthSession, loadCapabilities, loadOperatorStatus, loadPresets]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.code === "Space") {
        e.preventDefault();
        if (!panelUnavailable("trace")) void s.runSimulation();
      }
      if (e.key === "1") activatePanel("trace");
      if (e.key === "2" && hasPhase) activatePanel("phase");
      if (e.key === "3") activatePanel("fi-curve");
      if (e.key === "4") activatePanel("bifurcation");
      if (e.key === "5") activatePanel("sensitivity");
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  return (
    <div className="studio">
      <DevelopmentPreviewBanner
        deploymentProfile={s.operatorStatus?.deployment_profile ?? "development"}
      />
      <header className="header">
        <div className="header-logo">
          <div className="dot" />
          <h1>SC-NeuroCore Studio</h1>
        </div>
        <CapabilityStrip />
        <AuthControl />

        <div style={{ display: "flex", gap: 0, borderRadius: "var(--radius)", overflow: "hidden" }}>
          <Tab active={s.sourceMode === "model"} color="var(--accent)"
            label="Models" onClick={() => s.setSourceMode("model")} />
          <Tab active={s.sourceMode === "ode"} color="var(--warning)"
            label="ODE" onClick={() => s.setSourceMode("ode")} />
        </div>

        {s.sourceMode === "ode" && <TemplateLibrary />}

        <Btn label={s.isSimulating ? "..." : "Run"} onClick={s.runSimulation}
          disabled={s.isSimulating || panelUnavailable("trace")}
          title={panelState("trace").message} />
        <Btn label="Char." onClick={s.runCharacterize}
          disabled={s.isSimulating || s.sourceMode !== "model" || panelUnavailable("characterize")}
          title={panelState("characterize").message}
          color="#fff176" />
        <Btn label="f-I" onClick={s.runFICurve}
          disabled={s.isSimulating || panelUnavailable("fi-curve")}
          title={panelState("fi-curve").message}
          color="var(--success)" />
        <Btn label="Sens" onClick={s.runSensitivity}
          disabled={s.isSimulating || panelUnavailable("sensitivity")}
          title={panelState("sensitivity").message}
          color="#ce93d8" />
        <Btn label="Code" onClick={s.runCodegen}
          disabled={panelUnavailable("code")}
          title={panelState("code").message}
          color="#90a4ae" />

        {paramKeys.length > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
            <select value={s.sweepParam} onChange={(e) => s.setSweepParam(e.target.value)}
              style={{ fontSize: 9, padding: "1px 2px", maxWidth: 70 }}>
              <option value="">X...</option>
              {paramKeys.map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
            <Btn label="Bif" onClick={s.runBifurcation}
              disabled={s.isSimulating || !s.sweepParam || panelUnavailable("bifurcation")}
              title={panelState("bifurcation").message}
              color="#ef9a9a" />
            <select value={s.sweepParamY} onChange={(e) => s.setSweepParamY(e.target.value)}
              style={{ fontSize: 9, padding: "1px 2px", maxWidth: 70 }}>
              <option value="">Y...</option>
              {paramKeys.filter((k) => k !== s.sweepParam).map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
            <Btn label="2D" onClick={s.runHeatmap}
              disabled={s.isSimulating || !s.sweepParam || !s.sweepParamY || panelUnavailable("heatmap")}
              title={panelState("heatmap").message}
              color="#ffab91" />
          </div>
        )}

        {s.sourceMode === "ode" && (
          <>
            <Btn label="Q8.8" onClick={s.runPrecision}
              disabled={s.isSimulating || panelUnavailable("precision")}
              title={panelState("precision").message}
              color="#80deea" />
            <Btn label="RTL" onClick={s.runCompile}
              disabled={s.isSimulating || panelUnavailable("verilog")}
              title={panelState("verilog").message}
              color="#a5d6a7" />
            <Btn label="IR" onClick={s.runBuildIR}
              disabled={s.isSimulating || panelUnavailable("ir")}
              title={panelState("ir").message}
              color="#ffcc80" />
            <Btn label="SV" onClick={s.runEmitSV}
              disabled={s.isSimulating || panelUnavailable("ir")}
              title={panelState("ir").message}
              color="#c5e1a5" />
          </>
        )}

        <Btn label="Canvas" onClick={() => activatePanel("canvas")}
          {...panelControl("canvas")}
          color="#4fc3f7" />
        <Btn label="Train" onClick={() => activatePanel("train")}
          {...panelControl("train")}
          color="#b39ddb" />
        <Btn label="Admin" onClick={() => activatePanel("admin")}
          {...panelControl("admin")}
          color="#ffcc80" />
        <Btn label="E-I Net" onClick={s.runNetwork}
          disabled={s.isSimulating || panelUnavailable("network")}
          title={panelState("network").message}
          color="#80cbc4" />
        <Btn label="STA" onClick={s.computeSTA} disabled={!s.result || s.result.spikes.length < 3} color="#b0bec5" />
        <Btn label="Freq" onClick={s.runFreqResponse}
          disabled={s.isSimulating || panelUnavailable("freq")}
          title={panelState("freq").message}
          color="#fff176" />
        {s.sourceMode === "ode" && s.equations.length >= 2 && (
          <Btn label="Nullcl." onClick={s.runNullclines}
            disabled={s.isSimulating || panelUnavailable("sensitivity")}
            title={panelState("sensitivity").message}
            color="#ef9a9a" />
        )}
        <Btn label="Import" onClick={() => {
          const csv = prompt("Paste voltage trace (one value per line, or CSV):");
          if (csv) s.importCSV(csv);
        }} outline />
        <Btn label="Share" onClick={s.shareURL} outline />
        <Btn label="Reset" onClick={s.resetDefaults} outline />
        <Btn label="JSON" onClick={s.exportData} disabled={!s.result} outline />
        <Btn label="CSV" onClick={s.exportCSV} disabled={!s.result} outline />
        <Btn label="SVG" onClick={s.exportSVG} disabled={!s.result} outline />

        <div className="header-spacer" />

        <div style={{ display: "flex", gap: 0, borderRadius: "var(--radius)", flexWrap: "wrap" }}>
          <Tab active={s.activeTab === "trace"} color="var(--accent)" label="Trace" onClick={() => activatePanel("trace")} {...panelControl("trace")} />
          {hasPhase && <Tab active={s.activeTab === "phase"} color="#ce93d8" label="Phase" onClick={() => activatePanel("phase")} {...panelControl("phase")} />}
          {hasISI && <Tab active={s.activeTab === "isi"} color="var(--warning)" label="ISI" onClick={() => activatePanel("isi")} {...panelControl("isi")} />}
          <Tab active={s.activeTab === "fi-curve"} color="var(--success)" label="f-I" onClick={() => activatePanel("fi-curve")} {...panelControl("fi-curve")} />
          <Tab active={s.activeTab === "bifurcation"} color="#ef9a9a" label="Bif" onClick={() => activatePanel("bifurcation")} {...panelControl("bifurcation")} />
          <Tab active={s.activeTab === "heatmap"} color="#ffab91" label="2D" onClick={() => activatePanel("heatmap")} {...panelControl("heatmap")} />
          <Tab active={s.activeTab === "sensitivity"} color="#ce93d8" label="Sens" onClick={() => activatePanel("sensitivity")} {...panelControl("sensitivity")} />
          <Tab active={s.activeTab === "sta"} color="#b0bec5" label="STA" onClick={() => activatePanel("sta")} {...panelControl("sta")} />
          <Tab active={s.activeTab === "freq"} color="#fff176" label="Freq" onClick={() => activatePanel("freq")} {...panelControl("freq")} />
          {s.sourceMode === "model" && (
            <Tab active={s.activeTab === "characterize"} color="#fff176" label="Char" onClick={() => activatePanel("characterize")} {...panelControl("characterize")} />
          )}
          <Tab active={s.activeTab === "multi"} color="#80cbc4" label="Multi" onClick={() => activatePanel("multi")} {...panelControl("multi")} />
          <Tab active={s.activeTab === "compare"} color="#ce93d8" label="A/B" onClick={() => activatePanel("compare")} {...panelControl("compare")} />
          <Tab active={s.activeTab === "network"} color="#80cbc4" label="E-I" onClick={() => activatePanel("network")} {...panelControl("network")} />
          <Tab active={s.activeTab === "code"} color="#90a4ae" label="Code" onClick={() => activatePanel("code")} {...panelControl("code")} />
          <Tab active={s.activeTab === "delays"} color="#f48fb1" label="Delays" onClick={() => activatePanel("delays")} {...panelControl("delays")} />
          {s.sourceMode === "ode" && (
            <>
              <Tab active={s.activeTab === "precision"} color="#80deea" label="Q8.8" onClick={() => activatePanel("precision")} {...panelControl("precision")} />
              <Tab active={s.activeTab === "verilog"} color="#a5d6a7" label="RTL" onClick={() => activatePanel("verilog")} {...panelControl("verilog")} />
              <Tab active={s.activeTab === "ir"} color="#ffcc80" label="IR" onClick={() => activatePanel("ir")} {...panelControl("ir")} />
              <Tab active={s.activeTab === "synth"} color="#a5d6a7" label="FPGA"
                onClick={() => activatePanel("synth")}
                {...panelControl("synth")} />
            </>
          )}
          <Tab active={s.activeTab === "canvas"} color="#4fc3f7" label="Canvas"
            onClick={() => activatePanel("canvas")}
            {...panelControl("canvas")} />
          <Tab active={s.activeTab === "train"} color="#b39ddb" label="Train" onClick={() => activatePanel("train")} {...panelControl("train")} />
          <Tab active={s.activeTab === "admin"} color="#ffcc80" label="Admin" onClick={() => activatePanel("admin")} {...panelControl("admin")} />
        </div>

        {pattern && (
          <span className="header-stats" style={{
            color: pattern.pattern === "tonic" ? "var(--success)" :
                   pattern.pattern === "bursting" ? "var(--warning)" :
                   pattern.pattern === "silent" ? "var(--text-muted)" : "var(--accent)",
          }}>
            {pattern.pattern} {s.result ? `${s.result.stats.rate_hz}Hz` : ""}
          </span>
        )}
      </header>

      {s.error && <div className="error-banner">{s.error}</div>}

      {s.isSimulating && s.progressMsg && (
        <div style={{
          padding: "4px 16px", borderBottom: "1px solid var(--border)",
          background: "var(--bg-secondary)",
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 2 }}>
            <span style={{ color: "var(--text-secondary)" }}>{s.progressMsg}</span>
            <span style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{s.progressPct}%</span>
          </div>
          <div style={{ height: 4, background: "var(--bg-tertiary)", borderRadius: 2, overflow: "hidden" }}>
            <div style={{
              height: "100%", width: `${s.progressPct}%`, background: "var(--accent)",
              borderRadius: 2, transition: "width 0.3s",
            }} />
          </div>
        </div>
      )}

      {s.codeOneliner && (
        <div style={{
          padding: "4px 16px", fontSize: 10, fontFamily: "var(--font-mono)",
          background: "var(--bg-secondary)", borderBottom: "1px solid var(--border)",
          color: "var(--text-muted)", cursor: "pointer", overflow: "hidden", whiteSpace: "nowrap",
          textOverflow: "ellipsis",
        }} onClick={() => { navigator.clipboard.writeText(s.codeOneliner); }}
          title="Click to copy">
          {s.codeOneliner}
        </div>
      )}

      <div className="main-content">
        <div className="left-panel">
          <div className="panel-section">
            <EvidenceCartStrip
              cart={evidenceCart}
              error={evidenceCartError}
              lastExport={evidenceCartExport}
              onExport={() => {
                void exportSessionEvidenceCart().catch(() => undefined);
              }}
            />
          </div>
          <div className="panel-section">
            <OperatorWorkbenchPanel
              onExportEvidence={operateWorkbenchEvidenceBundle}
              onOpenAdmin={() => activatePanel("admin")}
              onOpenCompiler={() => activatePanel(s.sourceMode === "ode" ? "verilog" : "canvas")}
              onOpenProjects={() => activatePanel(s.sourceMode === "model" ? "trace" : "ir")}
              onRunSimulation={() => {
                if (!s.isSimulating && !panelUnavailable("trace")) {
                  void s.runSimulation().then(() => {
                    queueSimulationIntoEvidenceCart();
                  });
                }
              }}
              state={operatorWorkbench}
            />
          </div>
          <div className="panel-section">
            <StudioReadinessPanel
              model={studioReadiness}
              onOpenAdmin={() => activatePanel("admin")}
              onRefresh={refreshStudioReadiness}
            />
          </div>
          <div className="panel-section">
            <GuidedFlowPanel controller={guidedRun} state={guidedFlow} />
          </div>
          {s.sourceMode === "model" ? (
            <>
              <div className="panel-section">
                <div className="panel-header">Model Library ({s.models.length})</div>
                <ModelBrowser />
              </div>
              <MultiModelPicker />
              <ModelComparison />
            </>
          ) : (
            <div className="panel-section">
              <div className="panel-header">Equations</div>
              <EquationEditor />
            </div>
          )}

          {s.presets.length > 0 && (
            <div className="panel-section">
              <div className="panel-header">Experiments ({s.presets.length})</div>
              <div style={{ maxHeight: 100, overflowY: "auto" }}>
                {s.presets.map((p) => (
                  <div key={p.id} onClick={() => s.loadPreset(p.id)} style={{
                    padding: "2px 6px", fontSize: 10, cursor: "pointer",
                    borderRadius: 3, color: "var(--text-secondary)",
                  }} title={p.description}>
                    {p.title}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="panel-section">
            <div className="panel-header">Projects</div>
            <div style={{ display: "flex", gap: 4, marginBottom: 4 }}>
              <button aria-label="Save project" onClick={() => {
                const name = prompt("Project name:");
                if (name) s.saveProjectToServer(name);
              }} style={{
                fontSize: 10, padding: "2px 6px", background: "var(--bg-tertiary)",
                color: "var(--text-secondary)", border: "1px solid var(--border)",
                borderRadius: 3, cursor: "pointer",
              }}>Save</button>
              <button aria-label="Refresh projects" onClick={() => s.listServerProjects()} style={{
                fontSize: 10, padding: "2px 6px", background: "var(--bg-tertiary)",
                color: "var(--text-secondary)", border: "1px solid var(--border)",
                borderRadius: 3, cursor: "pointer",
              }}>Refresh</button>
            </div>
            {s.serverProjects.length > 0 && (
              <div style={{ maxHeight: 60, overflowY: "auto" }}>
                {s.serverProjects.map((p) => (
                  <div key={p.name} style={{
                    display: "flex", justifyContent: "space-between", fontSize: 10,
                    padding: "1px 4px", color: "var(--text-secondary)",
                  }}>
                    <span style={{ cursor: "pointer" }} onClick={() => s.loadProjectFromServer(p.name)}>{p.name}</span>
                    <span style={{ cursor: "pointer", color: "var(--text-muted)" }}
                      onClick={() => s.deleteServerProject(p.name)}>x</span>
                  </div>
                ))}
              </div>
            )}
            {s.projectSaveResult && (
              <ProjectEvidenceStrip
                artifacts={s.projectEvidenceBundle?.artifacts ?? []}
                evidence={buildProjectEvidenceModel(s.projectSaveResult)}
                exportBundleId={s.projectEvidenceBundle?.bundle_id ?? null}
                exportError={s.projectEvidenceBundleError}
                exportJobId={s.projectEvidenceBundle?.job_id ?? null}
                loading={s.projectEvidenceBundleLoading}
                onDownloadArtifact={(relativePath) => void s.downloadEvidenceBundleArtifactForSurface("project", relativePath)}
                onExportBundle={exportProjectEvidenceBundle}
              />
            )}
          </div>

          <div className="panel-section">
            <div className="panel-header">Sessions</div>
            <div style={{ display: "flex", gap: 4, marginBottom: 4 }}>
              <button aria-label="Save session" onClick={() => {
                const name = prompt("Session name:");
                if (name) s.saveSession(name);
              }} style={{
                fontSize: 10, padding: "2px 6px", background: "var(--bg-tertiary)",
                color: "var(--text-secondary)", border: "1px solid var(--border)",
                borderRadius: 3, cursor: "pointer",
              }}>Save</button>
            </div>
            {s.savedSessions.length > 0 && (
              <div style={{ maxHeight: 60, overflowY: "auto" }}>
                {s.savedSessions.map((ss) => (
                  <div key={ss.name} style={{
                    display: "flex", justifyContent: "space-between", fontSize: 10,
                    padding: "1px 4px", color: "var(--text-secondary)",
                  }}>
                    <span style={{ cursor: "pointer" }} onClick={() => s.loadSession(ss.name)}>{ss.name}</span>
                    <span style={{ cursor: "pointer", color: "var(--text-muted)" }}
                      onClick={() => s.deleteSession(ss.name)}>x</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="panel-section">
            <div className="panel-header">Info</div>
            <ModelInfo />
            {pattern && (
              <div style={{
                marginTop: 4, fontSize: 11,
                color: pattern.pattern === "tonic" ? "var(--success)" :
                       pattern.pattern === "bursting" ? "var(--warning)" :
                       pattern.pattern === "silent" ? "var(--text-muted)" : "var(--text-primary)",
              }}>
                {pattern.description}
              </div>
            )}
          </div>

          {s.sourceMode === "model" && s.modelDetail && (
            <BackendMatrix backends={s.modelDetail.backends} />
          )}

          <ModelDocViewer />

          {s.result && (
            <div className="panel-section">
              <div className="panel-header">Spike Statistics</div>
              <SpikeStats />
            </div>
          )}

          <ComparePanel />
          <NetworkControls />
          <ParameterSliders />
        </div>

        <div className="right-panel">
          {!activePanelState.available ? (
            <CapabilityUnavailable state={activePanelState} />
          ) : s.activeTab === "canvas" ? (
            <NetworkCanvas />
          ) : s.activeTab === "train" ? (
            <TrainingMonitor />
          ) : s.activeTab === "admin" ? (
            <AdminPanel />
          ) : s.activeTab === "synth" ? (
            <SynthesisDashboard />
          ) : s.activeTab === "ir" ? (
            <CompilerInspector />
          ) : s.activeTab === "verilog" ? (
            <VerilogPreview />
          ) : s.activeTab === "delays" ? (
            <DclsPanel />
          ) : s.activeTab === "code" ? (
            <div style={{ flex: 1, padding: 8 }}>
              {s.codeScript ? (
                <pre style={{
                  fontSize: 11, fontFamily: "var(--font-mono)",
                  color: "var(--text-primary)", background: "var(--bg-secondary)",
                  padding: 12, borderRadius: "var(--radius)",
                  overflow: "auto", height: "100%", whiteSpace: "pre-wrap",
                }}>
                  {s.codeScript}
                </pre>
              ) : (
                <div style={{ color: "var(--text-muted)", padding: 20 }}>
                  Click "Code" to generate a Python script
                </div>
              )}
            </div>
          ) : (
            <SimulationPlot />
          )}
        </div>
      </div>
      <StatusBar />
      <KeyboardHelp />
      <OnboardingOverlay />
    </div>
  );
}
