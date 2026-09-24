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

## What kind of identity each entry is

`total` counts every registered identity. It is **not** the number of models
from the literature, and a browser that showed only a total invited exactly that
reading. Each entry now carries:

| Field | Meaning |
| --- | --- |
| `identity_kind` | `source-literature`, `project-original`, `sc-compatibility`, or `api-alias` |
| `counts_in_source_catalogue` | Whether it counts towards the public source catalogue |
| `public_label` | The label it is published under, empty when the registry holds none |
| `aliases` | Other names that resolve to this identity |

`GET /api/models/facets` adds `identity_kinds`, the census by kind, and
`source_catalogue_total`, which is smaller than `total`. At the time of writing
the corpus is 133 from the literature, 27 project originals and 25
SC-compatibility identities, of which 160 count towards the source catalogue.

The classification comes from `neurons/model_identity.py`, the same registry the
identity ledger is generated from — not a second count kept beside it. An API
alias is an identity there and not a registered catalogue model, so it reaches
no row and cannot inflate the literature count.

## Querying the catalogue

The catalogue grows as models are enrolled, so the filtering rule lives on the
server, in one place. `GET /api/models/query` takes any of these parameters;
each one left out is no filter:

| Parameter | Admits |
| --- | --- |
| `text` | Models whose name, public label, aliases, family or summary contain it (case-insensitive) |
| `family`, `behavior`, `identity_kind`, `metadata_state` | Models with that value |
| `min_verified_science` | Models proven at that science tier or higher (0–5) |
| `min_verified_silicon` | Models proven at that silicon tier or higher (0–5); a model not enrolled on silicon meets no floor above 0 |
| `verified_perfect_only` | `true`: models proven at S5 and at their declared terminal silicon tier |

The readiness floors read **only the verified tiers**, the ones bound to facet
receipts whose subjects still match the repository. A descriptor that declares
S5 and is proven at S2 is not "at least S3". Each catalogue entry carries
`is_perfect_verified`, the same judgement the model detail makes, so no client
has to fetch every detail to filter on it.

The answer (`sc-neurocore.studio.catalogue-query.v1`) names the matching
models, `matched` and `total`, the `corpus_revision` it was computed on, and
facet counts. `family`, `behavior`, `identity_kind` and `metadata_state` are
counted over the models every *other* filter admits, so choosing a family still
shows how many models each other family would give; the verified-tier counts
and `verified_perfect` are counted over the matched models. An unknown
parameter, a tier outside 0–5 or a non-boolean `verified_perfect_only` is
refused with `422` and `detail.reason`.

The model browser sends its search text, family, behaviour and readiness
filters to this route and lists what it admits. The only filter it applies
itself is a firing pattern from its own live scan, which exists nowhere else.
Its readiness filters are labelled "proven" and are buttons a keyboard reaches
and a screen reader hears as pressed or not.

## Linking to a model

`#model=<ClassName>` at the end of the Studio's address opens that model, for
example `…/studios/sc-neurocore/#model=AdExNeuron`. The link names only the
catalogue identity, in plain text; unlike a share link it carries no run
settings and does not change them. **Copy link** in the model panel builds it.
A link naming a model this catalogue does not hold (renamed or removed since
the link was made) opens nothing and says which name failed.

## Guards

`tests/test_studio_catalogue_metadata_state.py` makes one descriptor load raise
and asserts what follows: the listing keeps its length, the entry is present and
marked `invalid` with its diagnostic, the facet census names it, the corpus
revision moves, an invalid entry carries the same keys as a healthy one, and
the runtime-state conformance scope still contains the broken model.
