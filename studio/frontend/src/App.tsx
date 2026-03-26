import { useStudioStore } from "./stores/studio";
import TemplateLibrary from "./components/TemplateLibrary";
import EquationEditor from "./components/EquationEditor";
import ParameterSliders from "./components/ParameterSliders";
import SimulationPlot from "./components/SimulationPlot";
import ModelInfo from "./components/ModelInfo";
import SpikeStats from "./components/SpikeStats";

export default function App() {
  const {
    runSimulation, runFICurve, exportData,
    isSimulating, error, result, activeTab, setActiveTab,
  } = useStudioStore();

  return (
    <div className="studio">
      <header className="header">
        <div className="header-logo">
          <div className="dot" />
          <h1>SC-NeuroCore Studio</h1>
        </div>
        <TemplateLibrary />
        <button className="btn-simulate" onClick={runSimulation} disabled={isSimulating}>
          {isSimulating ? "..." : "Simulate"}
        </button>
        <button className="btn-simulate" onClick={runFICurve} disabled={isSimulating}
          style={{ background: "#81c784" }}>
          f-I Curve
        </button>
        <button className="btn-simulate" onClick={exportData}
          disabled={!result}
          style={{ background: "transparent", border: "1px solid #30363d", color: "#8b949e" }}>
          Export
        </button>

        <div style={{ display: "flex", gap: 2, marginLeft: 8 }}>
          <button
            className="btn-simulate"
            onClick={() => setActiveTab("trace")}
            style={{
              background: activeTab === "trace" ? "#4fc3f7" : "transparent",
              color: activeTab === "trace" ? "#0d1117" : "#8b949e",
              border: "1px solid #30363d",
              borderRadius: "4px 0 0 4px",
              padding: "4px 10px", fontSize: 12,
            }}
          >Trace</button>
          <button
            className="btn-simulate"
            onClick={() => setActiveTab("fi-curve")}
            style={{
              background: activeTab === "fi-curve" ? "#81c784" : "transparent",
              color: activeTab === "fi-curve" ? "#0d1117" : "#8b949e",
              border: "1px solid #30363d",
              borderRadius: "0 4px 4px 0",
              padding: "4px 10px", fontSize: 12,
            }}
          >f-I</button>
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
          <div className="panel-section">
            <div className="panel-header">Equations</div>
            <EquationEditor />
          </div>
          <div className="panel-section">
            <div className="panel-header">Model</div>
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
          <SimulationPlot />
        </div>
      </div>
    </div>
  );
}
