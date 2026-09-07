# Catalogue health

The Studio catalogue lists **every registered model**, including a model whose
metadata cannot be read. It says which ones those are.

## Why a listing that never shrinks

The catalogue used to drop an entry whose metadata raised while it was being
built. Nothing recorded the drop. A corrupt descriptor therefore showed up as a
catalogue with one fewer model in it — a smaller success count where a fault
belonged.

The cost was not confined to the browser. Several surfaces derive their scope
from the catalogue listing, `tools/runtime_state_conformance.py` among them.
A model that fell out of the listing fell out of the conformance matrix with
it, so the gate stopped checking the model that had just broken and reported
nothing.

## Per-entry metadata state

Each entry carries a `metadata_state` and, when it is not readable, a
`metadata_error`:

| State | Meaning |
| --- | --- |
| `available` | A committed descriptor was loaded. |
| `unavailable` | The model is real but has no committed descriptor; the entry is built from code introspection, and `category_source` reads `inferred`. |
| `invalid` | The metadata could not be read. `metadata_error` names the failure; the entry keeps every field a healthy entry has, with declared-unknown values. |

An `invalid` entry stays in the listing and stays in every consumer's scope. It
is browsable as a fault, not absent.

## Corpus health in the facets

`GET /api/models/facets` reports:

- `total` — every registered identity. It does not move when a descriptor
  breaks, because nothing was removed.
- `metadata_states` — how many entries are in each state.
- `invalid_models` — the unreadable entries, by name.
- `corpus_revision` — a digest over the identities and their states. Two
  clients holding the same revision hold the same corpus in the same health;
  the digest changes when a model is added or removed *and* when any entry's
  state changes, so a degraded corpus never shares a revision with a healthy
  one of the same size.

In the model browser, a degraded corpus raises a notice naming the affected
models, and each affected row carries its state inline.

## Guards

`tests/test_studio_catalogue_metadata_state.py` makes one descriptor load raise
and asserts what follows: the listing keeps its length, the entry is present and
marked `invalid` with its diagnostic, the facet census names it, the corpus
revision moves, an invalid entry carries the same keys as a healthy one, and
the runtime-state conformance scope still contains the broken model.
