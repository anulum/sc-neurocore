<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Studio OpenAPI reference -->

# Studio OpenAPI contract

The Studio backend exposes its HTTP contract through FastAPI at
`/openapi.json`. The version-controlled
[OpenAPI 3.1 document](../_generated/studio_openapi.json) is generated from the
same application factory used by the CLI and production server.

The contract includes every public path, method, operation identifier,
parameter, request body, response schema, and validation schema. CI-facing
checks compare the committed document with the runtime application so route or
schema drift cannot land silently.

The conditionally mounted frontend document at `/` is a UI delivery route, not
an API operation, and is excluded from the schema. This keeps the reference
identical whether or not the ignored production `studio/frontend/dist/` bundle
exists in the checkout.

## Responsibility routers

| Responsibility | Route family |
| --- | --- |
| System and capabilities | `/api/health`, `/api/studio/capabilities*`, operator status |
| Jobs | `/api/studio/jobs*` |
| Audit and evidence | `/api/studio/audit*`, `/api/studio/evidence/bundle` |
| Identity | `/api/studio/auth*`, `/api/studio/identity*` |
| Catalogue and presets | `/api/models*`, `/api/templates*`, `/api/presets*` |
| Simulation and analysis | simulation, analysis, characterisation, network routes |
| Compile and co-simulate | `/api/compile`, `/api/nir/compile`, `/api/ir/*` |
| Synthesis and hardware handoff | `/api/synth*`, `/api/pipeline/run` |
| Design and training | project, graph, training, export, and progress routes |

Regenerate after an intentional API change:

```bash
PYTHONPATH=src:. .venv/bin/python tools/generate_studio_openapi.py
```

Verify without modifying the committed reference:

```bash
PYTHONPATH=src:. .venv/bin/python tools/generate_studio_openapi.py --check
```
