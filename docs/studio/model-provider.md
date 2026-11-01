<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Model discovery provider guide -->

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

Bundled model schemas are available through
`sc_neurocore.neurons.model_identity.SCHEMA_DIR`; parse them with the public
`sc_neurocore.neurons.universal_dsl.load_schema` function. Studio uses this
resource directory to hash the selected schema into an experiment's identity,
preferring TOML over JSON when both formats exist. Installed workers read the
schemas shipped with their library; no neighbouring source checkout is needed.

The public seed contract lives in `sc_neurocore.neurons.seed_domain`.
`DEFAULT_LFSR16_SEED` exposes the canonical deterministic fallback `0xACE1`
(44257); `LFSR16_ALIASES` maps an explicit LFSR16 request of zero to this state.
Studio equation simulations use the same default when `seed` is omitted, and
an authored equation IR request serializes its omitted `rng_seed` as the exact
string `"44257"`. Explicit counter-based random keys remain distinct: zero is
not an alias for 44257 in `rand()`, `randn()`, `poisson()` or `xi` draws.
Model constructors that accept `None` for fresh entropy keep that behaviour.
