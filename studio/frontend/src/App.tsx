import { useStudioStore } from "./stores/studio";
import TemplateLibrary from "./components/TemplateLibrary";
import EquationEditor from "./components/EquationEditor";
import ParameterSliders from "./components/ParameterSliders";
import SimulationPlot from "./components/SimulationPlot";

export default function App() {
  const { runSimulation, isSimulating, error, result } = useStudioStore();

  return (
    <div
      style={{
        fontFamily: "system-ui, sans-serif",
        background: "#121212",
        color: "#e0e0e0",
        minHeight: "100vh",
        padding: 20,
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 16,
          marginBottom: 16,
        }}
      >
        <h1 style={{ margin: 0, fontSize: 20 }}>SC-NeuroCore Studio</h1>
        <TemplateLibrary />
        <button
          onClick={runSimulation}
          disabled={isSimulating}
          style={{
            padding: "6px 16px",
            fontSize: 14,
            background: isSimulating ? "#555" : "#4fc3f7",
            color: "#000",
            border: "none",
            borderRadius: 4,
            cursor: isSimulating ? "wait" : "pointer",
          }}
        >
          {isSimulating ? "Running..." : "Simulate"}
        </button>
        {result && (
          <span style={{ fontSize: 13, color: "#888" }}>
            {result.spike_count} spikes in {result.n_steps} steps
          </span>
        )}
      </header>

      {error && (
        <div
          style={{
            background: "#421c1c",
            padding: "8px 12px",
            borderRadius: 4,
            marginBottom: 12,
            fontSize: 13,
          }}
        >
          {error}
        </div>
      )}

      <div style={{ display: "flex", gap: 20 }}>
        <div style={{ flex: "0 0 380px" }}>
          <EquationEditor />
          <div style={{ marginTop: 12 }}>
            <ParameterSliders />
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <SimulationPlot />
        </div>
      </div>
    </div>
  );
}
