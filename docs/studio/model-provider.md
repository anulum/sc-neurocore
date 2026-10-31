# Model discovery for Studio integrations

Use the public read-only registry to discover model class identifiers and their
implementation module basenames without loading every model:

```python
import importlib

from sc_neurocore.neurons.model_registry import MODEL_MODULES

name = "AdExNeuron"
module = importlib.import_module(
    f"sc_neurocore.neurons.models.{MODEL_MODULES[name]}"
)
neuron = getattr(module, name)()
spike = neuron.step(0.0)
```

The mapping retains declaration order, includes every catalogue identity and
excludes historical import-only aliases. It cannot be changed by assignment or
deletion. Unknown identifiers raise `KeyError` on lookup; model import failures
propagate instead of silently removing a catalogue entry. Reading the mapping
does not import model implementations or require their optional runtime backends.

Studio uses this same provider for catalogue metadata, constructor admission,
network population validation and catalogue-to-candidate preparation. It does
not maintain its own copy of the model list. The provider ships with the base
library, so installed Studio workers use the same contract as standalone users.

Registry membership says that a class is registered. Scientific classification,
parameter contracts, numerical profiles and verified hardware readiness come
from their separate descriptor, identity, profile and evidence APIs. A historical
class alias that remains importable does not become a new catalogue identity.
