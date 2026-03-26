import { useStudioStore } from "../stores/studio";

export default function ModelInfo() {
  const { equations, params, init, dt, duration } = useStudioStore();
  const nSteps = Math.min(Math.floor(duration / dt), 100_000);

  return (
    <div className="model-info">
      <div className="info-item">
        <span className="info-label">eqs:</span>
        <span className="info-value">{equations.length}</span>
      </div>
      <div className="info-item">
        <span className="info-label">vars:</span>
        <span className="info-value">{Object.keys(init).join(", ")}</span>
      </div>
      <div className="info-item">
        <span className="info-label">params:</span>
        <span className="info-value">{Object.keys(params).length}</span>
      </div>
      <div className="info-item">
        <span className="info-label">steps:</span>
        <span className="info-value">{nSteps.toLocaleString()}</span>
      </div>
    </div>
  );
}
