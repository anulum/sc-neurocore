import { useEffect } from "react";
import { useStudioStore } from "./stores/studio";
import TemplateLibrary from "./components/TemplateLibrary";
import EquationEditor from "./components/EquationEditor";
import ParameterSliders from "./components/ParameterSliders";
import SimulationPlot from "./components/SimulationPlot";
import ModelInfo from "./components/ModelInfo";
import SpikeStats from "./components/SpikeStats";
import ModelBrowser from "./components/ModelBrowser";
import VerilogPreview from "./components/VerilogPreview";
import MultiModelPicker from "./components/MultiModelPicker";
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

function Tab({ active, color, label, onClick }: {
  active: boolean; color: string; label: string; onClick: () => void;
}) {
  return (
    <button onClick={onClick} style={{
      padding: "2px 6px", fontSize: 9, fontWeight: 600,
      fontFamily: "var(--font-ui)", lineHeight: 1.4,
      background: active ? color : "transparent",
      color: active ? "var(--bg-primary)" : "var(--text-muted)",
      border: "1px solid var(--border)", cursor: "pointer", whiteSpace: "nowrap",
    }}>{label}</button>
  );
}

function Btn({ label, onClick, disabled, color, outline }: {
  label: string; onClick: () => void; disabled?: boolean;
  color?: string; outline?: boolean;
}) {
  return (
    <button className="btn-simulate" onClick={onClick} disabled={disabled} style={{
      background: outline ? "transparent" : (color || "var(--accent)"),
      border: outline ? "1px solid var(--border)" : "none",
      color: outline ? "var(--text-muted)" : "var(--bg-primary)",
      padding: "2px 7px", fontSize: 10,
    }}>{label}</button>
  );
}

export default function App() {
  const s = useStudioStore();
  const loadCapabilities = s.loadCapabilities;
  const loadPresets = s.loadPresets;
  const vars = s.result ? Object.keys(s.result.states) : [];
  const hasPhase = vars.length >= 2;
  const hasISI = s.result?.stats?.isi_histogram != null;
  const paramKeys = Object.keys(s.sourceMode === "model" ? s.modelParams : s.odeParams);
  const pattern = s.result?.pattern;

  useEffect(() => {
    void loadCapabilities();
    void loadPresets();
  }, [loadCapabilities, loadPresets]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.code === "Space") { e.preventDefault(); s.runSimulation(); }
      if (e.key === "1") s.setActiveTab("trace");
      if (e.key === "2" && hasPhase) s.setActiveTab("phase");
      if (e.key === "3") s.setActiveTab("fi-curve");
      if (e.key === "4") s.setActiveTab("bifurcation");
      if (e.key === "5") s.setActiveTab("sensitivity");
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  return (
    <div className="studio">
      <header className="header">
        <div className="header-logo">
          <div className="dot" />
          <h1>SC-NeuroCore Studio</h1>
        </div>
        <CapabilityStrip />

        <div style={{ display: "flex", gap: 0, borderRadius: "var(--radius)", overflow: "hidden" }}>
          <Tab active={s.sourceMode === "model"} color="var(--accent)"
            label="Models" onClick={() => s.setSourceMode("model")} />
          <Tab active={s.sourceMode === "ode"} color="var(--warning)"
            label="ODE" onClick={() => s.setSourceMode("ode")} />
        </div>

        {s.sourceMode === "ode" && <TemplateLibrary />}

        <Btn label={s.isSimulating ? "..." : "Run"} onClick={s.runSimulation} disabled={s.isSimulating} />
        <Btn label="Char." onClick={s.runCharacterize}
          disabled={s.isSimulating || s.sourceMode !== "model"} color="#fff176" />
        <Btn label="f-I" onClick={s.runFICurve} disabled={s.isSimulating} color="var(--success)" />
        <Btn label="Sens" onClick={s.runSensitivity} disabled={s.isSimulating} color="#ce93d8" />
        <Btn label="Code" onClick={s.runCodegen} color="#90a4ae" />

        {paramKeys.length > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
            <select value={s.sweepParam} onChange={(e) => s.setSweepParam(e.target.value)}
              style={{ fontSize: 9, padding: "1px 2px", maxWidth: 70 }}>
              <option value="">X...</option>
              {paramKeys.map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
            <Btn label="Bif" onClick={s.runBifurcation} disabled={s.isSimulating || !s.sweepParam} color="#ef9a9a" />
            <select value={s.sweepParamY} onChange={(e) => s.setSweepParamY(e.target.value)}
              style={{ fontSize: 9, padding: "1px 2px", maxWidth: 70 }}>
              <option value="">Y...</option>
              {paramKeys.filter((k) => k !== s.sweepParam).map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
            <Btn label="2D" onClick={s.runHeatmap}
              disabled={s.isSimulating || !s.sweepParam || !s.sweepParamY} color="#ffab91" />
          </div>
        )}

        {s.sourceMode === "ode" && (
          <>
            <Btn label="Q8.8" onClick={s.runPrecision} disabled={s.isSimulating} color="#80deea" />
            <Btn label="RTL" onClick={s.runCompile} disabled={s.isSimulating} color="#a5d6a7" />
            <Btn label="IR" onClick={s.runBuildIR} disabled={s.isSimulating} color="#ffcc80" />
            <Btn label="SV" onClick={s.runEmitSV} disabled={s.isSimulating} color="#c5e1a5" />
          </>
        )}

        <Btn label="Canvas" onClick={() => s.setActiveTab("canvas")} color="#4fc3f7" />
        <Btn label="Train" onClick={() => s.setActiveTab("train")} color="#b39ddb" />
        <Btn label="E-I Net" onClick={s.runNetwork} disabled={s.isSimulating} color="#80cbc4" />
        <Btn label="STA" onClick={s.computeSTA} disabled={!s.result || s.result.spikes.length < 3} color="#b0bec5" />
        <Btn label="Freq" onClick={s.runFreqResponse} disabled={s.isSimulating} color="#fff176" />
        {s.sourceMode === "ode" && s.equations.length >= 2 && (
          <Btn label="Nullcl." onClick={s.runNullclines} disabled={s.isSimulating} color="#ef9a9a" />
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

        <div style={{ display: "flex", gap: 0, borderRadius: "var(--radius)", overflow: "hidden" }}>
          <Tab active={s.activeTab === "trace"} color="var(--accent)" label="Trace" onClick={() => s.setActiveTab("trace")} />
          {hasPhase && <Tab active={s.activeTab === "phase"} color="#ce93d8" label="Phase" onClick={() => s.setActiveTab("phase")} />}
          {hasISI && <Tab active={s.activeTab === "isi"} color="var(--warning)" label="ISI" onClick={() => s.setActiveTab("isi")} />}
          <Tab active={s.activeTab === "fi-curve"} color="var(--success)" label="f-I" onClick={() => s.setActiveTab("fi-curve")} />
          <Tab active={s.activeTab === "bifurcation"} color="#ef9a9a" label="Bif" onClick={() => s.setActiveTab("bifurcation")} />
          <Tab active={s.activeTab === "heatmap"} color="#ffab91" label="2D" onClick={() => s.setActiveTab("heatmap")} />
          <Tab active={s.activeTab === "sensitivity"} color="#ce93d8" label="Sens" onClick={() => s.setActiveTab("sensitivity")} />
          <Tab active={s.activeTab === "sta"} color="#b0bec5" label="STA" onClick={() => s.setActiveTab("sta")} />
          <Tab active={s.activeTab === "freq"} color="#fff176" label="Freq" onClick={() => s.setActiveTab("freq")} />
          {s.sourceMode === "model" && (
            <Tab active={s.activeTab === "characterize"} color="#fff176" label="Char" onClick={() => s.setActiveTab("characterize")} />
          )}
          <Tab active={s.activeTab === "multi"} color="#80cbc4" label="Multi" onClick={() => s.setActiveTab("multi")} />
          <Tab active={s.activeTab === "compare"} color="#ce93d8" label="A/B" onClick={() => s.setActiveTab("compare")} />
          <Tab active={s.activeTab === "network"} color="#80cbc4" label="E-I" onClick={() => s.setActiveTab("network")} />
          <Tab active={s.activeTab === "code"} color="#90a4ae" label="Code" onClick={() => s.setActiveTab("code")} />
          {s.sourceMode === "ode" && (
            <>
              <Tab active={s.activeTab === "precision"} color="#80deea" label="Q8.8" onClick={() => s.setActiveTab("precision")} />
              <Tab active={s.activeTab === "verilog"} color="#a5d6a7" label="RTL" onClick={() => s.setActiveTab("verilog")} />
              <Tab active={s.activeTab === "ir"} color="#ffcc80" label="IR" onClick={() => s.setActiveTab("ir")} />
              <Tab active={s.activeTab === "synth"} color="#a5d6a7" label="FPGA" onClick={() => s.setActiveTab("synth")} />
            </>
          )}
          <Tab active={s.activeTab === "canvas"} color="#4fc3f7" label="Canvas" onClick={() => s.setActiveTab("canvas")} />
          <Tab active={s.activeTab === "train"} color="#b39ddb" label="Train" onClick={() => s.setActiveTab("train")} />
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
          {s.sourceMode === "model" ? (
            <>
              <div className="panel-section">
                <div className="panel-header">Model Library (118)</div>
                <ModelBrowser />
              </div>
              <MultiModelPicker />
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
              <button onClick={() => {
                const name = prompt("Project name:");
                if (name) s.saveProjectToServer(name);
              }} style={{
                fontSize: 10, padding: "2px 6px", background: "var(--bg-tertiary)",
                color: "var(--text-secondary)", border: "1px solid var(--border)",
                borderRadius: 3, cursor: "pointer",
              }}>Save</button>
              <button onClick={() => s.listServerProjects()} style={{
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
          </div>

          <div className="panel-section">
            <div className="panel-header">Sessions</div>
            <div style={{ display: "flex", gap: 4, marginBottom: 4 }}>
              <button onClick={() => {
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
          {s.activeTab === "canvas" ? (
            <NetworkCanvas />
          ) : s.activeTab === "train" ? (
            <TrainingMonitor />
          ) : s.activeTab === "synth" ? (
            <SynthesisDashboard />
          ) : s.activeTab === "ir" ? (
            <CompilerInspector />
          ) : s.activeTab === "verilog" ? (
            <VerilogPreview />
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
