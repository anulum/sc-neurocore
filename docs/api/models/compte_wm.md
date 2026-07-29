# CompteWMNeuron

**Module:** sc_neurocore.neurons.models.compte_wm
**Source:** Compte, Brunel, Goldman-Rakic & Wang, *Cerebral Cortex* 10(9),
910–923 (2000)
**DOI:** 10.1093/cercor/10.9.910
**Scope:** excitatory pyramidal cell and incoming AMPA/NMDA/GABAA kinetics

## Scientific identity

CompteWMNeuron implements the source paper's pyramidal leaky
integrate-and-fire cell together with the incoming channel-state equations. It
is not the complete 2,560-cell spatial working-memory network.

The public input pathways are deliberately separate:

- external_spike=True increments the external AMPA gate;
- spike_in=True increments the recurrent NMDA precursor. The historical
  argument name is retained for compatibility;
- inhibitory_spike=True increments the incoming GABAA gate;
- current is direct somatic current in nA.

An output spike resets the membrane and starts the refractory interval. It does
not increment GABA: the former self-inhibition rule was not present in the
cited model.

## Equations

The membrane equation is

$$
C_m\frac{dV}{dt} =
-g_L(V-E_L)
-g_{AMPA}s_{AMPA}(V-E_{exc})
-g_{NMDA}B(V)s_{NMDA}(V-E_{exc})
-g_{GABA}s_{GABA}(V-E_{inh})
+I.
$$

The Jahr–Stevens magnesium-unblock factor is

$$
B(V)=\frac{1}{1+[Mg^{2+}]\exp(-0.062V)/3.57}.
$$

Incoming channel kinetics are

$$
\frac{ds_{AMPA}}{dt}=-\frac{s_{AMPA}}{\tau_{AMPA}},
$$

$$
\frac{ds_{NMDA}}{dt}=-\frac{s_{NMDA}}{\tau_{NMDA}}
+\alpha x_{NMDA}(1-s_{NMDA}),
$$

$$
\frac{dx_{NMDA}}{dt}=-\frac{x_{NMDA}}{\tau_x},
\qquad
\frac{ds_{GABA}}{dt}=-\frac{s_{GABA}}{\tau_{GABA}}.
$$

Presynaptic events add one to the corresponding AMPA, NMDA-precursor, or
GABAA state before the continuous flow. The five continuous variables advance
together with explicit midpoint RK2 at 0.02 ms.

Threshold detection is sampled after the step:

$$
V_{candidate}\ge V_{threshold}
\Rightarrow V\leftarrow V_{reset},\quad
t_{ref}\leftarrow\tau_{ref}.
$$

The paper used firing-time interpolation. This implementation does not claim
equivalence to that within-step timing scheme.

## Source control-set defaults

Conductances use µS, voltages mV, current nA, capacitance nF, and time ms.

| Parameter | Default | Source meaning |
|---|---:|---|
| g_l | 0.025 µS | 25 nS pyramidal leak |
| g_ampa | 0.0031 µS | 3.1 nS external pyramidal AMPA |
| g_nmda | 0.000381 µS | 0.381 nS recurrent pyramidal NMDA |
| g_gaba | 0.001336 µS | 1.336 nS interneuron→pyramidal GABAA |
| e_l, e_inh | −70 mV | leak and inhibitory reversal |
| e_exc | 0 mV | AMPA/NMDA reversal |
| c_m | 0.5 nF | pyramidal capacitance |
| mg | 1 mM | extracellular magnesium |
| tau_ampa | 2 ms | AMPA decay |
| tau_nmda | 100 ms | NMDA open-fraction decay |
| tau_x | 2 ms | NMDA rise-precursor decay |
| tau_gaba | 10 ms | GABAA decay |
| alpha_nmda | 0.5 ms⁻¹ | NMDA saturation rate |
| v_threshold | −50 mV | sampled threshold |
| v_reset | −60 mV | pyramidal reset |
| tau_ref | 2 ms | pyramidal absolute refractory time |
| dt | 0.02 ms | source integration step |

These conductances select the paper's control-set pathways for one pyramidal
cell. Network weights and the connectivity footprint remain external.

## Python API

    from sc_neurocore.neurons.models.compte_wm import CompteWMNeuron

    cell = CompteWMNeuron()
    event = cell.step(
        current=1.0,
        spike_in=True,          # recurrent NMDA event
        external_spike=True,   # external AMPA event
        inhibitory_spike=False,
    )
    state = cell.get_state()

get_state() returns v, s_ampa, s_nmda, x_nmda, s_gaba, and ref_remaining.
reset() clears those dynamic values while preserving every configuration
field. Invalid mutable state, configuration, current, or native input fails
without partial state mutation.

The complete batch API is:

    result = cell.simulate(
        currents,
        recurrent_nmda_events,
        external_ampa_events,
        inhibitory_gabaa_events,
        backend="auto",
    )

Each event array contains only zero or one. The result contains six complete
state traces, the sampled output-event vector, and all six final-state values.

## Executed backends

The maintained dispatcher exposes real Python, modular Rust/PyO3, Julia, Go,
and Mojo implementations. The 200,000-step committed benchmark records 812
events in every lane. Python, Rust, Julia, and Go are binary64-identical for
the complete receipt; Mojo remains within 3.6e-14, below the declared 2e-10
tolerance. These are local loaded-host regression measurements, not
exclusive-core production-speed claims.

Native API documentation lives with each declaration:

- Python docstrings on the class, step, batch, reset, and state surfaces;
- Rustdoc on the modular engine, PyO3 boundary, and independent safety kernel;
- GoDoc on exported state, validation, step, reset, and service functions;
- Julia docstrings on the public module/type/step/batch/reset surfaces;
- Mojo ABI and helper doc comments at the exported declarations;
- RTL interface and fixed-point-contract comments at the module.

## Source, schema, and silicon evidence

The independent 1,024-step primary-equation receipt pins all six state values
and the four output events at indices 276, 505, 736, and 971. Its canonical
row digest is
bc071ae0c0057bc23a5e4c99ee5bbee53d306f07cd154e01bafff32335a6e192.

Paired TOML and JSON schemas implement a nine-edge lowering protocol and match
the hand model. The Q16.16 RTL preserves the complete enrolled 1,024-step event
vector, keeps voltage within 0.35 mV and each channel-state error within the
declared bounds, synthesizes in Yosys, and passes a depth-4 CVC5 bounded safety
job.

This is H1 evidence only. It does not establish binary64 formal equivalence,
firing-time interpolation, timing, PPA, device behavior, or production
deployment.

## Retained SC network successor

The 2,560-cell ring, columnar connectivity footprint, Poisson drive,
persistent bump, distractor resistance, and network statistics are retained
for a separately named SC project-derived model. They are not deleted, folded
into this cell, or promoted by this cell's evidence. That successor must earn
its own source/specification, parity, benchmark, and silicon evidence.
