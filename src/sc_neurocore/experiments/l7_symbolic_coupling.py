from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

from ..sources.bitstream_current_source import BitstreamCurrentSource
from ..neurons.stochastic_lif import StochasticLIFNeuron
from ..recorders.spike_recorder import BitstreamSpikeRecorder

# Add HolonomicAtlas simulations to path for cross-layer engine import
for parent in Path(__file__).resolve().parents:
    candidate = parent / "SCPN-CODEBASE"
    if candidate.exists():
        holon_sim = candidate / "HolonomicAtlas" / "Simulations"
        sys.path.insert(0, str(holon_sim))
        break

from scpn_cross_layer_integration_L1_L7 import CrossLayerIntegrationEngine_L1_L7

GLYPH_FEATURE_ORDER = [
    "phi_alignment",
    "fibonacci_alignment",
    "metatron_flow",
    "platonic_coherence",
    "e8_alignment",
    "symbolic_health",
]

FEATURE_SCALE_MAP = {
    "phi_alignment": 1.0,
    "fibonacci_alignment": 1.0,
    "metatron_flow": 6.0,
    "platonic_coherence": 1.0,
    "e8_alignment": 1.0,
    "symbolic_health": 1.0,
}

GLYPH_WEIGHT_MAP = {
    "phi_alignment": 0.82,
    "fibonacci_alignment": 0.78,
    "metatron_flow": 0.95,
    "platonic_coherence": 0.74,
    "e8_alignment": 0.7,
    "symbolic_health": 0.65,
}


def gather_symbolic_features() -> tuple[list[float], dict[str, float]]:
    engine = CrossLayerIntegrationEngine_L1_L7()
    engine.run(duration=0.5, dt=0.001)
    last_state = engine.history["L7_states"][-1]

    raw_diagnostics = {
        "phi_alignment": float(last_state.phi_alignment),
        "fibonacci_alignment": float(last_state.fibonacci_alignment),
        "metatron_flow": float(last_state.metatron_flow),
        "platonic_coherence": float(last_state.platonic_coherence),
        "e8_alignment": float(last_state.e8_alignment),
        "symbolic_health": float(last_state.symbolic_health),
    }

    normalized_features = []
    for key in GLYPH_FEATURE_ORDER:
        scale = FEATURE_SCALE_MAP.get(key, 1.0)
        value = raw_diagnostics[key] / scale
        normalized_features.append(float(np.clip(value, 0.0, 1.0)))

    print("L7 glyph diagnostics (golden ratio, Fibonacci, Metatron, Platonic, E8, health):")
    for key, value in raw_diagnostics.items():
        print(f"  {key}: {value:.4f}")
    print("Normalized glyph feature vector:", normalized_features)

    return normalized_features, raw_diagnostics


def run():
    features, diagnostics = gather_symbolic_features()
    glyph_weights = [GLYPH_WEIGHT_MAP[name] for name in GLYPH_FEATURE_ORDER]
    print("Applying glyph weights (phi, Fibonacci, Metatron, Platonic, E8, health):", glyph_weights)

    source = BitstreamCurrentSource(
        x_inputs=features,
        x_min=0.0,
        x_max=1.0,
        weight_values=glyph_weights,
        w_min=0.0,
        w_max=1.0,
        length=8192,
        y_min=0.0,
        y_max=0.3,
        seed=42,
    )

    neuron = StochasticLIFNeuron(
        v_rest=0.0,
        v_reset=0.0,
        v_threshold=1.0,
        tau_mem=20.0,
        dt=1.0,
        noise_std=0.02,
        resistance=1.0,
        seed=1234,
    )

    recorder = BitstreamSpikeRecorder(dt_ms=neuron.dt)
    T = 2000
    for _ in range(T):
        I_t = source.step()
        spike = neuron.step(I_t)
        recorder.record(spike)

    print("Total spikes:", recorder.total_spikes())
    print("Firing rate (Hz):", recorder.firing_rate_hz())
    hist, edges = recorder.isi_histogram(bins=10)
    print("ISI histogram counts:", hist)
    print("ISI bin edges (ms):", edges)


if __name__ == "__main__":
    run()
