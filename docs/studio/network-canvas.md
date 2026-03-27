# Network Canvas

The Network Canvas provides a visual drag-and-drop interface for
designing spiking neural networks. Add excitatory and inhibitory
populations, connect them with projections, configure weights and
connectivity, then simulate — all from a browser.

Built on React Flow (Xyflow), the canvas supports node dragging,
edge creation via drag-connect, and real-time layout updates.

## Quick Start

1. Switch to the **Canvas** tab
2. Click **+ Exc** to add an excitatory population
3. Click **+ Inh** to add an inhibitory population
4. Drag from one node's handle to another to create a projection
5. Click **Simulate** to run the network
6. View spike counts, firing rates in the results bar

## Populations

Each population is a group of identical neurons:

| Property | Default | Description |
|----------|---------|-------------|
| label | auto | Display name (e.g., "Exc 0") |
| model | LIFNeuron | Neuron model from the 118 available |
| count | 80 (exc) / 20 (inh) | Number of neurons |
| neuron_type | excitatory | Excitatory (blue) or inhibitory (red) |
| position | auto | Canvas x, y coordinates |

Excitatory populations are shown with rounded blue borders.
Inhibitory populations use square red borders.

## Projections

Projections are directed connections between populations:

| Property | Default | Description |
|----------|---------|-------------|
| weight | 0.1 | Synaptic weight (positive for exc, negated for inh→exc) |
| delay | 1.0 ms | Synaptic delay |
| probability | 0.2 | Connection probability (sparse random connectivity) |

Create projections by dragging from a source node handle to a target
node handle. The weight label appears on the edge.

## Simulation

The canvas maps populations and projections to the E-I balanced
network simulation backend. For networks with excitatory and
inhibitory populations:

- Exc→Exc weight = w_ee
- Exc→Inh weight = w_ie
- Inh→Exc weight = w_ei
- Inh→Inh weight = w_ii

Connection probability from projections is used for sparse random
connectivity.

**Limits:** Maximum 2000 neurons per network (browser performance).

Results appear in the status bar: neuron count, spike count, mean
excitatory and inhibitory firing rates.

## NIR Export/Import

The canvas supports NIR (Neuromorphic Intermediate Representation)
format for interoperability with other SNN frameworks:

- **Export NIR:** Saves populations as nodes and projections as edges
  in a JSON file compatible with the NIR specification
- **Import NIR:** Loads a NIR JSON file and creates populations and
  projections on the canvas

### NIR Format

```json
{
  "format": "nir",
  "version": "0.1",
  "nodes": {
    "pop_a": {"type": "LIF", "count": 80, "neuron_type": "excitatory"},
    "pop_b": {"type": "LIF", "count": 20, "neuron_type": "inhibitory"}
  },
  "edges": [
    {"source": "pop_a", "target": "pop_b", "weight": 0.5, "delay": 1.0}
  ]
}
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/graph/models` | List available neuron models |
| POST | `/api/graph/population` | Create a population node |
| POST | `/api/graph/projection` | Create a projection edge |
| POST | `/api/graph/validate` | Validate network graph |
| POST | `/api/graph/simulate` | Simulate network |
| POST | `/api/graph/export-nir` | Export to NIR format |
| POST | `/api/graph/import-nir` | Import from NIR format |

### POST /api/graph/population

```json
{"label": "Exc 0", "model": "LIFNeuron", "count": 80, "neuron_type": "excitatory", "x": 100, "y": 100}
```

### POST /api/graph/simulate

```json
{
  "populations": [...],
  "projections": [...],
  "duration": 200.0,
  "dt": 0.1
}
```

Returns spike times, neuron indices, firing rates, and a graph summary.

### POST /api/graph/validate

Returns `{"valid": true, "errors": []}` or validation errors:
- Missing populations
- Dangling projection endpoints
- Zero-weight projections
- Neuron count exceeding 2000 limit
- Probability out of (0, 1] range
