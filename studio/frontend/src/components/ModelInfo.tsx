import { useStudioStore } from "../stores/studio";

export default function ModelInfo() {
  const { sourceMode, modelDetail, equations, odeParams, odeInit, dt, duration } = useStudioStore();

  const nSteps = Math.min(Math.floor(duration / dt), 100_000);

  if (sourceMode === "model" && modelDetail) {
    return (
      <div>
        <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4 }}>
          {modelDetail.docstring || modelDetail.name}
        </div>
        <div className="model-info">
          <div className="info-item">
            <span className="info-label">vars:</span>
            <span className="info-value">
              {modelDetail.state_vars.map((s) => s.name).join(", ")}
            </span>
          </div>
          <div className="info-item">
            <span className="info-label">params:</span>
            <span className="info-value">{modelDetail.params.length}</span>
          </div>
          <div className="info-item">
            <span className="info-label">dt:</span>
            <span className="info-value">{modelDetail.dt}</span>
          </div>
          <div className="info-item">
            <span className="info-label">steps:</span>
            <span className="info-value">{nSteps.toLocaleString()}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="model-info">
      <div className="info-item">
        <span className="info-label">eqs:</span>
        <span className="info-value">{equations.length}</span>
      </div>
      <div className="info-item">
        <span className="info-label">vars:</span>
        <span className="info-value">{Object.keys(odeInit).join(", ")}</span>
      </div>
      <div className="info-item">
        <span className="info-label">params:</span>
        <span className="info-value">{Object.keys(odeParams).length}</span>
      </div>
      <div className="info-item">
        <span className="info-label">steps:</span>
        <span className="info-value">{nSteps.toLocaleString()}</span>
      </div>
    </div>
  );
}
