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
| `<izhikevich2007Cell>` | SCIzhikevichNeuron | Biophysical units, converted |
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

### Izhikevich (2007 -> 2003)

NeuroML `<izhikevich2007Cell>` uses physical units (pF, mV, nS).
SC-NeuroCore uses the dimensionless 2003 model. Parameters are
approximately mapped. The raw 2007 values are preserved in
`params["_neuroml2007_raw"]` for reference.

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
