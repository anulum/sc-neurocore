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

function Tab({ active, color, label, onClick }: {
  active: boolean; color: string; label: string; onClick: () => void;
}) {
  return (
    <button onClick={onClick} style={{
      padding: "3px 7px", fontSize: 10, fontWeight: 600,
      fontFamily: "var(--font-ui)", lineHeight: 1.4,
      background: active ? color : "transparent",
      color: active ? "var(--bg-primary)" : "var(--text-muted)",
      border: "1px solid var(--border)", cursor: "pointer",
      whiteSpace: "nowrap",
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
      padding: "3px 8px", fontSize: 10,
    }}>{label}</button>
  );
}

export default function App() {
  const s = useStudioStore();
  const vars = s.result ? Object.keys(s.result.states) : [];
  const hasPhase = vars.length >= 2;
  const hasISI = s.result?.stats?.isi_histogram != null;
  const paramKeys = Object.keys(s.sourceMode === "model" ? s.modelParams : s.odeParams);

  useEffect(() => { s.loadPresets(); }, []);

  return (
    <div className="studio">
      <header className="header">
        <div className="header-logo">
          <div className="dot" />
          <h1>SC-NeuroCore Studio</h1>
        </div>

        <div style={{ display: "flex", gap: 0, borderRadius: "var(--radius)", overflow: "hidden" }}>
          <Tab active={s.sourceMode === "model"} color="var(--accent)"
            label="Models (118)" onClick={() => s.setSourceMode("model")} />
          <Tab active={s.sourceMode === "ode"} color="var(--warning)"
            label="Custom ODE" onClick={() => s.setSourceMode("ode")} />
        </div>

        {s.sourceMode === "ode" && <TemplateLibrary />}

        <Btn label={s.isSimulating ? "..." : "Run"} onClick={s.runSimulation} disabled={s.isSimulating} />
        <Btn label="f-I" onClick={s.runFICurve} disabled={s.isSimulating} color="var(--success)" />
        <Btn label="Sens." onClick={s.runSensitivity} disabled={s.isSimulating} color="#ce93d8" />

        {paramKeys.length > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <select value={s.sweepParam} onChange={(e) => s.setSweepParam(e.target.value)}
              style={{ fontSize: 10, padding: "2px 4px" }}>
              <option value="">Bifurc. param...</option>
              {paramKeys.map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
            <Btn label="Bifurc." onClick={s.runBifurcation} disabled={s.isSimulating || !s.sweepParam} color="#ef9a9a" />
          </div>
        )}

        {s.sourceMode === "ode" && (
          <>
            <Btn label="Q8.8" onClick={s.runPrecision} disabled={s.isSimulating} color="#80deea" />
            <Btn label="RTL" onClick={s.runCompile} disabled={s.isSimulating} color="#a5d6a7" />
          </>
        )}

        <Btn label="Reset" onClick={s.resetDefaults} outline />
        <Btn label="JSON" onClick={s.exportData} disabled={!s.result} outline />
        <Btn label="PNG" onClick={s.exportSVG} outline />

        <div className="header-spacer" />

        <div style={{ display: "flex", gap: 0, borderRadius: "var(--radius)", overflow: "hidden" }}>
          <Tab active={s.activeTab === "trace"} color="var(--accent)" label="Trace" onClick={() => s.setActiveTab("trace")} />
          {hasPhase && <Tab active={s.activeTab === "phase"} color="#ce93d8" label="Phase" onClick={() => s.setActiveTab("phase")} />}
          {hasISI && <Tab active={s.activeTab === "isi"} color="var(--warning)" label="ISI" onClick={() => s.setActiveTab("isi")} />}
          <Tab active={s.activeTab === "fi-curve"} color="var(--success)" label="f-I" onClick={() => s.setActiveTab("fi-curve")} />
          <Tab active={s.activeTab === "bifurcation"} color="#ef9a9a" label="Bif." onClick={() => s.setActiveTab("bifurcation")} />
          <Tab active={s.activeTab === "sensitivity"} color="#ce93d8" label="Sens." onClick={() => s.setActiveTab("sensitivity")} />
          {s.sourceMode === "ode" && (
            <>
              <Tab active={s.activeTab === "precision"} color="#80deea" label="Q8.8" onClick={() => s.setActiveTab("precision")} />
              <Tab active={s.activeTab === "verilog"} color="#a5d6a7" label="RTL" onClick={() => s.setActiveTab("verilog")} />
            </>
          )}
        </div>

        {s.result && (
          <span className="header-stats">
            {s.result.stats.rate_hz}Hz {s.result.spike_count}spk
          </span>
        )}
      </header>

      {s.error && <div className="error-banner">{s.error}</div>}

      <div className="main-content">
        <div className="left-panel">
          {s.sourceMode === "model" ? (
            <div className="panel-section">
              <div className="panel-header">Model Library</div>
              <ModelBrowser />
            </div>
          ) : (
            <div className="panel-section">
              <div className="panel-header">Equations</div>
              <EquationEditor />
            </div>
          )}

          {s.presets.length > 0 && (
            <div className="panel-section">
              <div className="panel-header">Experiments</div>
              <div style={{ maxHeight: 120, overflowY: "auto" }}>
                {s.presets.map((p) => (
                  <div key={p.id} onClick={() => s.loadPreset(p.id)} style={{
                    padding: "3px 6px", fontSize: 11, cursor: "pointer",
                    borderRadius: 3, color: "var(--text-secondary)",
                  }} title={p.description}>
                    {p.title}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="panel-section">
            <div className="panel-header">Info</div>
            <ModelInfo />
          </div>

          {s.result && (
            <div className="panel-section">
              <div className="panel-header">Spike Statistics</div>
              <SpikeStats />
            </div>
          )}

          <ParameterSliders />
        </div>

        <div className="right-panel">
          {s.activeTab === "verilog" ? <VerilogPreview /> : <SimulationPlot />}
        </div>
      </div>
    </div>
  );
}
