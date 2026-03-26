import { useStudioStore } from "../stores/studio";

function sliderRange(value: number): [number, number, number] {
  const absVal = Math.abs(value) || 1;
  const lo = value - absVal * 2;
  const hi = value + absVal * 2;
  const step = absVal / 100;
  return [lo, hi, step];
}

function formatNum(n: number): string {
  if (Math.abs(n) >= 100) return n.toFixed(1);
  if (Math.abs(n) >= 1) return n.toPrecision(4);
  if (n === 0) return "0";
  return n.toPrecision(3);
}

export default function ParameterSliders() {
  const {
    params, init, current, dt, duration,
    setParam, setInit, setCurrent, setDt, setDuration,
  } = useStudioStore();

  return (
    <div>
      <div className="panel-section">
        <div className="panel-header">Parameters</div>
        {Object.entries(params).map(([key, value]) => {
          const [lo, hi, step] = sliderRange(value);
          return (
            <div className="slider-row" key={key}>
              <span className="slider-label">{key}</span>
              <input
                type="range" min={lo} max={hi} step={step} value={value}
                onChange={(e) => setParam(key, parseFloat(e.target.value))}
              />
              <span className="slider-value">{formatNum(value)}</span>
            </div>
          );
        })}
      </div>

      <div className="panel-section">
        <div className="panel-header">Initial State</div>
        {Object.entries(init).map(([key, value]) => {
          const [lo, hi, step] = sliderRange(value);
          return (
            <div className="slider-row" key={`init-${key}`}>
              <span className="slider-label">{key}₀</span>
              <input
                type="range" min={lo} max={hi} step={step} value={value}
                onChange={(e) => setInit(key, parseFloat(e.target.value))}
              />
              <span className="slider-value">{formatNum(value)}</span>
            </div>
          );
        })}
      </div>

      <div className="panel-section">
        <div className="panel-header">Stimulation</div>
        <div className="slider-row">
          <span className="slider-label">I</span>
          <input
            type="range" min={-100} max={100} step={0.1} value={current}
            onChange={(e) => setCurrent(parseFloat(e.target.value))}
          />
          <span className="slider-value">{formatNum(current)}</span>
        </div>
        <div className="slider-row">
          <span className="slider-label">dt</span>
          <input
            type="range" min={0.001} max={1} step={0.001} value={dt}
            onChange={(e) => setDt(parseFloat(e.target.value))}
          />
          <span className="slider-value">{dt}</span>
        </div>
        <div className="slider-row">
          <span className="slider-label">T (ms)</span>
          <input
            type="range" min={10} max={1000} step={10} value={duration}
            onChange={(e) => setDuration(parseFloat(e.target.value))}
          />
          <span className="slider-value">{duration}</span>
        </div>
      </div>
    </div>
  );
}
