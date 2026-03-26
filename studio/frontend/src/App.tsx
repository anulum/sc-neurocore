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
      padding: "3px 8px", fontSize: 10, fontWeight: 600,
      fontFamily: "var(--font-ui)", lineHeight: 1.4,
      background: active ? color : "transparent",
      color: active ? "var(--bg-primary)" : "var(--text-muted)",
      border: "1px solid var(--border)", cursor: "pointer",
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
      padding: "4px 10px", fontSize: 11,
    }}>{label}</button>
  );
}

export default function App() {
  const {
    sourceMode, setSourceMode, result,
    runSimulation, runFICurve, runCompile, exportData, resetDefaults,
    isSimulating, error, activeTab, setActiveTab,
  } = useStudioStore();

  const vars = result ? Object.keys(result.states) : [];
  const hasPhase = vars.length >= 2;
  const hasISI = result?.stats?.isi_histogram != null;

  return (
    <div className="studio">
      <header className="header">
        <div className="header-logo">
          <div className="dot" />
          <h1>SC-NeuroCore Studio</h1>
        </div>

        <div style={{ display: "flex", gap: 0, borderRadius: "var(--radius)", overflow: "hidden" }}>
          <Tab active={sourceMode === "model"} color="var(--accent)"
            label="Models (118)" onClick={() => setSourceMode("model")} />
          <Tab active={sourceMode === "ode"} color="var(--warning)"
            label="Custom ODE" onClick={() => setSourceMode("ode")} />
        </div>

        {sourceMode === "ode" && <TemplateLibrary />}

        <Btn label={isSimulating ? "..." : "Run"} onClick={runSimulation} disabled={isSimulating} />
        <Btn label="f-I" onClick={runFICurve} disabled={isSimulating} color="var(--success)" />
        {sourceMode === "ode" && (
          <Btn label="Compile" onClick={runCompile} disabled={isSimulating} color="#ce93d8" />
        )}
        <Btn label="Reset" onClick={resetDefaults} outline />
        <Btn label="Export" onClick={exportData} disabled={!result} outline />

        <div style={{ display: "flex", gap: 0, marginLeft: 4, borderRadius: "var(--radius)", overflow: "hidden" }}>
          <Tab active={activeTab === "trace"} color="var(--accent)" label="Trace" onClick={() => setActiveTab("trace")} />
          {hasPhase && <Tab active={activeTab === "phase"} color="#ce93d8" label="Phase" onClick={() => setActiveTab("phase")} />}
          {hasISI && <Tab active={activeTab === "isi"} color="var(--warning)" label="ISI" onClick={() => setActiveTab("isi")} />}
          <Tab active={activeTab === "fi-curve"} color="var(--success)" label="f-I" onClick={() => setActiveTab("fi-curve")} />
          {sourceMode === "ode" && <Tab active={activeTab === "verilog"} color="#ce93d8" label="RTL" onClick={() => setActiveTab("verilog")} />}
        </div>

        <div className="header-spacer" />
        {result && (
          <span className="header-stats">
            {result.stats.rate_hz} Hz &middot; {result.spike_count} spk &middot; {result.n_steps.toLocaleString()} steps
          </span>
        )}
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="main-content">
        <div className="left-panel">
          {sourceMode === "model" ? (
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

          <div className="panel-section">
            <div className="panel-header">Info</div>
            <ModelInfo />
          </div>

          {result && (
            <div className="panel-section">
              <div className="panel-header">Spike Statistics</div>
              <SpikeStats />
            </div>
          )}

          <ParameterSliders />
        </div>

        <div className="right-panel">
          {activeTab === "verilog" ? <VerilogPreview /> : <SimulationPlot />}
        </div>
      </div>
    </div>
  );
}
