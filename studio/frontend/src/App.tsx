import { useStudioStore } from "./stores/studio";
import TemplateLibrary from "./components/TemplateLibrary";
import EquationEditor from "./components/EquationEditor";
import ParameterSliders from "./components/ParameterSliders";
import SimulationPlot from "./components/SimulationPlot";
import ModelInfo from "./components/ModelInfo";

export default function App() {
  const { runSimulation, isSimulating, error, result } = useStudioStore();

  return (
    <div className="studio">
      <header className="header">
        <div className="header-logo">
          <div className="dot" />
          <h1>SC-NeuroCore Studio</h1>
        </div>
        <TemplateLibrary />
        <button
          className="btn-simulate"
          onClick={runSimulation}
          disabled={isSimulating}
        >
          {isSimulating ? "Running..." : "Simulate"}
        </button>
        <div className="header-spacer" />
        {result && (
          <span className="header-stats">
            {result.spike_count} spikes &middot; {result.n_steps.toLocaleString()} steps
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
          <ParameterSliders />
        </div>
        <div className="right-panel">
          <SimulationPlot />
        </div>
      </div>
    </div>
  );
}
