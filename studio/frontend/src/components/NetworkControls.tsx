import { useStudioStore } from "../stores/studio";
import type { StudioNetworkParams } from "../studioInputState";

type NetworkParamName = keyof StudioNetworkParams;

function NetSlider({ label, param, min, max, step }: {
  label: string; param: NetworkParamName; min: number; max: number; step: number;
}) {
  const { networkParams, setNetworkParam } = useStudioStore();
  const value = networkParams[param];
  return (
    <div className="slider-row">
      <span className="slider-label">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => setNetworkParam(param, parseFloat(e.target.value))} />
      <span className="slider-value">{value.toFixed(step < 0.1 ? 2 : 1)}</span>
    </div>
  );
}

export default function NetworkControls() {
  const { activeTab, runNetwork, isSimulating } = useStudioStore();
  if (activeTab !== "network") return null;
  return (
    <div className="panel-section">
      <div className="panel-header">E-I Network</div>
      <NetSlider label="N exc" param="n_exc" min={10} max={200} step={10} />
      <NetSlider label="N inh" param="n_inh" min={5} max={100} step={5} />
      <NetSlider label="w E→E" param="w_ee" min={0} max={1} step={0.01} />
      <NetSlider label="w E→I" param="w_ei" min={0} max={1} step={0.01} />
      <NetSlider label="w I→E" param="w_ie" min={0} max={1} step={0.01} />
      <NetSlider label="w I→I" param="w_ii" min={0} max={1} step={0.01} />
      <NetSlider label="p conn" param="p_conn" min={0.01} max={1} step={0.01} />
      <NetSlider label="ext Hz" param="ext_rate" min={0.1} max={100} step={0.5} />
      <button className="btn-simulate" onClick={runNetwork} disabled={isSimulating}
        style={{ width: "100%", marginTop: 4, background: "#80cbc4", color: "#0d1117", border: "none", padding: "3px 0", fontSize: 10 }}>
        {isSimulating ? "..." : "Run Network"}
      </button>
    </div>
  );
}
