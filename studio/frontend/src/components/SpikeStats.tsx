import { useStudioStore } from "../stores/studio";

export default function SpikeStats() {
  const { result } = useStudioStore();
  if (!result) return null;
  const { stats, spike_count } = result;

  return (
    <div className="model-info">
      <div className="info-item">
        <span className="info-label">spikes:</span>
        <span className="info-value">{spike_count}</span>
      </div>
      <div className="info-item">
        <span className="info-label">rate:</span>
        <span className="info-value">{stats.rate_hz} Hz</span>
      </div>
      {stats.isi_mean_ms !== null && (
        <div className="info-item">
          <span className="info-label">ISI:</span>
          <span className="info-value">{stats.isi_mean_ms} ms</span>
        </div>
      )}
      {stats.isi_cv !== null && (
        <div className="info-item">
          <span className="info-label">CV:</span>
          <span className="info-value">{stats.isi_cv}</span>
        </div>
      )}
    </div>
  );
}
