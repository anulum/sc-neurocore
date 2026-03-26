import { useStudioStore } from "../stores/studio";

function sliderRange(value: number): [number, number, number] {
  const absVal = Math.abs(value) || 1;
  const lo = value - absVal * 2;
  const hi = value + absVal * 2;
  const step = absVal / 100;
  return [lo, hi, step];
}

export default function ParameterSliders() {
  const { params, current, dt, duration, setParam, setCurrent, setDt, setDuration } =
    useStudioStore();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 13 }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>Parameters</div>
      {Object.entries(params).map(([key, value]) => {
        const [lo, hi, step] = sliderRange(value);
        return (
          <label key={key} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 70, textAlign: "right", fontFamily: "monospace" }}>
              {key}
            </span>
            <input
              type="range"
              min={lo}
              max={hi}
              step={step}
              value={value}
              onChange={(e) => setParam(key, parseFloat(e.target.value))}
              style={{ flex: 1 }}
            />
            <span style={{ width: 70, fontFamily: "monospace" }}>
              {value.toPrecision(4)}
            </span>
          </label>
        );
      })}
      <div style={{ fontWeight: 600, marginTop: 8, marginBottom: 4 }}>Simulation</div>
      <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ width: 70, textAlign: "right", fontFamily: "monospace" }}>I</span>
        <input
          type="range"
          min={-100}
          max={100}
          step={0.1}
          value={current}
          onChange={(e) => setCurrent(parseFloat(e.target.value))}
          style={{ flex: 1 }}
        />
        <span style={{ width: 70, fontFamily: "monospace" }}>
          {current.toPrecision(4)}
        </span>
      </label>
      <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ width: 70, textAlign: "right", fontFamily: "monospace" }}>dt</span>
        <input
          type="range"
          min={0.001}
          max={1}
          step={0.001}
          value={dt}
          onChange={(e) => setDt(parseFloat(e.target.value))}
          style={{ flex: 1 }}
        />
        <span style={{ width: 70, fontFamily: "monospace" }}>{dt}</span>
      </label>
      <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ width: 70, textAlign: "right", fontFamily: "monospace" }}>
          T (ms)
        </span>
        <input
          type="range"
          min={10}
          max={1000}
          step={10}
          value={duration}
          onChange={(e) => setDuration(parseFloat(e.target.value))}
          style={{ flex: 1 }}
        />
        <span style={{ width: 70, fontFamily: "monospace" }}>{duration}</span>
      </label>
    </div>
  );
}
