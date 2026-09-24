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

# Import all cells from a NeuroML file of point-cell definitions
cells = import_neuroml("my_cells.nml")

for cell in cells:
    print(f"{cell.cell_id}: {cell.cell_type}")
    for note in cell.notes:  # what the mapping assumed or could not carry
        print(f"  {note}")
    neuron = create_neuron(cell)
    spike = neuron.step(10.0)
```

## What Is Refused

The importer never invents a value. It raises `ValueError`, naming the cell and
attribute, when:

- an attribute the NeuroML schema requires is missing, including the cell `id`;
- a dimensional attribute is a bare number, or its unit has the wrong dimension
  (for example `C="100mV"`), or the value is not finite;
- a dimensionless attribute (the 2003 Izhikevich `a`, `b`, `c`, `d`) has a unit;
- the document holds an element it does not model -- a `network`, `population`,
  `projection`, `pulseGenerator`, ion channel or morphological `cell` -- because
  keeping only the cells would silently change what the document describes.

Documentation elements (`notes`, `annotation`, `property`) are ignored.

## Mapping Notes

Each `ImportedCell.notes` lists what its mapping assumed or could not carry:
the timestep the model runs at (NeuroML cells carry none), voltages taken
relative to the leak reversal, the normalised input resistance of the LIF
mapping, a refractory period rounded to whole timesteps, the 2003 Izhikevich
`v0` and `thresh` (the model has no place for them), and an AdEx `refract`
other than zero (`AdExNeuron` has no refractory period).

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
- Network topology (populations, projections) is not imported; a document that
  contains it is refused.
- Multi-compartment morphologies not supported.
- NeuroML is import-only: SC-NeuroCore has no NeuroML exporter.

## Validating Documents

The optional `neuroml` extra (`pip install "sc-neurocore[neuroml]"`) installs
libNeuroML, the reference library for writing NeuroML 2 documents and checking
them against the schema (`neuroml.utils.validate_neuroml2`). The importer's
tests import documents that libNeuroML wrote and validated.

## Reference

NeuroML 2 specification: https://docs.neuroml.org/Userdocs/Schemas/Cells.html
