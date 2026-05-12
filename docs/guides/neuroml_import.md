# NeuroML 2 Import Guide

Import neuron models from NeuroML 2 XML files into SC-NeuroCore.

## Supported Cell Types

| NeuroML 2 Tag | SC-NeuroCore Model | Notes |
|---------------|-------------------|-------|
| `<iafCell>` | StochasticLIFNeuron | Conductance-based, converted to tau-based |
| `<iafRefCell>` | StochasticLIFNeuron | With refractory period |
| `<iafTauCell>` | StochasticLIFNeuron | Direct tau mapping |
| `<iafTauRefCell>` | StochasticLIFNeuron | tau + refractory |
| `<izhikevichCell>` | SCIzhikevichNeuron | 2003 dimensionless model |
| `<izhikevich2007Cell>` | Izhikevich2007Neuron | Exact biophysical NeuroML units |
| `<adExIaFCell>` | AdExNeuron | Brette & Gerstner 2005 |

## Usage

```python
from sc_neurocore.adapters.neuroml import import_neuroml, create_neuron

# Import all cells from a NeuroML file
cells = import_neuroml("my_network.nml")

for cell in cells:
    print(f"{cell.cell_id}: {cell.cell_type}")
    neuron = create_neuron(cell)
    spike = neuron.step(10.0)
```

## Parameter Conversion

### LIF (iafCell -> StochasticLIFNeuron)

NeuroML specifies LIF with conductance and capacitance:
- `C` (pF), `leakConductance` (nS), `leakReversal` (mV)

SC-NeuroCore uses time constant:
- `tau_mem = C / leakConductance` (ms)
- Voltages normalised relative to leak reversal

### Izhikevich 2007

NeuroML `<izhikevich2007Cell>` uses physical units and is imported as
`Izhikevich2007Neuron` without conversion to the dimensionless 2003 model:

- `C` in pF, `k` in nS/mV, voltages in mV
- `a` in 1/ms, `b` in nS, `d` and runtime input current in pA
- RK4 integration is selected by default for imported cells

### AdEx

Direct parameter mapping. Units match (pF, nS, mV, ms).

## Limitations

- HH neurons (`<cell>` with `<biophysicalProperties>`) not yet supported.
  These require parsing ion channel definitions separately.
- Network topology (populations, projections) not imported.
  Use `import_neuroml()` for cell types only.
- Multi-compartment morphologies not supported.

## Reference

NeuroML 2 specification: https://docs.neuroml.org/Userdocs/Schemas/Cells.html
