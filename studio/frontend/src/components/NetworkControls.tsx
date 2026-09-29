// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useStudioStore } from "../stores/studio";
import type { StudioNetworkParams } from "../studioInputState";

/** The name of one balanced-network parameter. */
type NetworkParamName = keyof StudioNetworkParams;

/**
 * One labelled slider bound to a network parameter.
 *
 * @param props - The label, the parameter, and its bounds.
 * @returns The slider.
 */
function NetSlider({ label, name, unit, param, min, max, step }: {
  label: string; name: string; unit: string; param: NetworkParamName;
  min: number; max: number; step: number;
}) {
  const { networkParams, setNetworkParam } = useStudioStore();
  const value = networkParams[param];
  const shown = value.toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0);
  return (
    <div className="slider-row">
      <span className="slider-label" aria-hidden="true">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        aria-label={name} aria-valuetext={unit === "" ? shown : `${shown} ${unit}`}
        onChange={(e) => { setNetworkParam(param, parseFloat(e.target.value)); }} />
      <span className="slider-value">{shown}{unit === "" ? "" : ` ${unit}`}</span>
    </div>
  );
}

/**
 * The balanced-network parameters and its run button.
 *
 * @returns The controls.
 */
export default function NetworkControls() {
  const { activeTab, runNetwork, isSimulating } = useStudioStore();
  if (activeTab !== "network") return null;
  return (
    <div className="panel-section">
      <div className="panel-header">E-I network</div>
      <NetSlider label="N exc" name="Excitatory neurons" unit="" param="n_exc" min={10} max={200} step={10} />
      <NetSlider label="N inh" name="Inhibitory neurons" unit="" param="n_inh" min={5} max={100} step={5} />
      {/* Weights are membrane jumps in mV, named post-pre as the model does:
          w_ei is inhibitory-to-excitatory. The E→I and I→E labels were swapped. */}
      <NetSlider label="w E→E" name="Weight excitatory to excitatory" unit="mV" param="w_ee" min={0} max={1} step={0.01} />
      <NetSlider label="w E→I" name="Weight excitatory to inhibitory" unit="mV" param="w_ie" min={0} max={1} step={0.01} />
      <NetSlider label="w I→E" name="Weight inhibitory to excitatory" unit="mV" param="w_ei" min={0} max={1} step={0.01} />
      <NetSlider label="w I→I" name="Weight inhibitory to inhibitory" unit="mV" param="w_ii" min={0} max={1} step={0.01} />
      <NetSlider label="p conn" name="Connection probability" unit="" param="p_conn" min={0.01} max={1} step={0.01} />
      <NetSlider label="ext rate" name="Rate of each external input" unit="Hz" param="ext_rate" min={0} max={50} step={0.5} />
      <p className="panel-note">
        Each neuron receives 800 external inputs of 0.1 mV; about 9.4 Hz per input
        reaches threshold.
      </p>
      <button type="button" className="btn-simulate btn btn--primary" onClick={() => { void runNetwork(); }}
        disabled={isSimulating} style={{ width: "100%", marginTop: 4 }}>
        {isSimulating ? "Running…" : "Run E-I network"}
      </button>
    </div>
  );
}
