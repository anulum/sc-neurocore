import { useStudioStore } from "./stores/studio";
import TemplateLibrary from "./components/TemplateLibrary";
import EquationEditor from "./components/EquationEditor";
import ParameterSliders from "./components/ParameterSliders";
import SimulationPlot from "./components/SimulationPlot";
import ModelInfo from "./components/ModelInfo";
import SpikeStats from "./components/SpikeStats";
import ModelBrowser from "./components/ModelBrowser";
import VerilogPreview from "./components/VerilogPreview";

function TabButton({ active, color, label, onClick }: {
  active: boolean; color: string; label: string; onClick: () => void;
}) {
  return (
    <button onClick={onClick} style={{
      padding: "4px 10px", fontSize: 11, fontWeight: 600,
      fontFamily: "var(--font-ui)",
      background: active ? color : "transparent",
      color: active ? "var(--bg-primary)" : "var(--text-muted)",
      border: "1px solid var(--border)",
      cursor: "pointer",
    }}>{label}</button>
  );
}

export default function App() {
  const {
    sourceMode, setSourceMode,
    runSimulation, runFICurve, runCompile, exportData,
    isSimulating, error, result, activeTab, setActiveTab,
  } = useStudioStore();

  return (
    <div className="studio">
      <header className="header">
        <div className="header-logo">
          <div className="dot" />
          <h1>SC-NeuroCore Studio</h1>
        </div>

        <div style={{ display: "flex", gap: 0, borderRadius: "var(--radius)", overflow: "hidden" }}>
          <TabButton active={sourceMode === "model"} color="var(--accent)"
            label={`Models (118)`} onClick={() => setSourceMode("model")} />
          <TabButton active={sourceMode === "ode"} color="var(--warning)"
            label="Custom ODE" onClick={() => setSourceMode("ode")} />
        </div>

        {sourceMode === "ode" && <TemplateLibrary />}

        <button className="btn-simulate" onClick={runSimulation} disabled={isSimulating}>
          {isSimulating ? "..." : "Simulate"}
        </button>
        <button className="btn-simulate" onClick={runFICurve} disabled={isSimulating}
          style={{ background: "var(--success)" }}>f-I</button>
        {sourceMode === "ode" && (
          <button className="btn-simulate" onClick={runCompile} disabled={isSimulating}
            style={{ background: "#ce93d8" }}>Compile</button>
        )}
        <button className="btn-simulate" onClick={exportData} disabled={!result}
          style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--text-muted)" }}>
          Export
        </button>

        <div style={{ display: "flex", gap: 0, marginLeft: 8, borderRadius: "var(--radius)", overflow: "hidden" }}>
          <TabButton active={activeTab === "trace"} color="var(--accent)"
            label="Trace" onClick={() => setActiveTab("trace")} />
          <TabButton active={activeTab === "fi-curve"} color="var(--success)"
            label="f-I" onClick={() => setActiveTab("fi-curve")} />
          {sourceMode === "ode" && (
            <TabButton active={activeTab === "verilog"} color="#ce93d8"
              label="Verilog" onClick={() => setActiveTab("verilog")} />
          )}
        </div>

        <div className="header-spacer" />
        {result && (
          <span className="header-stats">
            {result.stats.rate_hz} Hz &middot; {result.n_steps.toLocaleString()} steps
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
            <div className="panel-header">
              {sourceMode === "model" ? "Model" : "ODE"} Info
            </div>
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
