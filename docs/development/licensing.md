<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore - Public licensing and compliance page
-->

# Licensing and Compliance

## 1) Legal basis for redistribution and use

SC-NeuroCore is licensed under **AGPL-3.0-or-later**.
Commercial licensing is available.

Commercial licensing terms do not remove AGPL-3.0-or-later obligations for AGPL deployments.
A separate commercial agreement must be obtained for non-AGPL commercial terms.

For default AGPL use, redistribution and deployment follow the standard AGPL terms and the standard network-service availability requirements:

- provide the corresponding source for modified public software,
- ensure recipients can obtain the source of modified versions,
- keep notices and license files intact.

## 2) Network and service deployments

For software or services that expose a modified deployment over a network,
the AGPL source-availability rule applies to all modified SC-NeuroCore components.
This is the network-service and source-availability boundary for modified server deployments.
Any modified code must be made available under the same licence terms in those cases.

Commercial licensing for network deployment can add commercial rights or support,
but it does not remove AGPL obligations for AGPL-classified code paths.
Commercial deployment terms only apply where a separate commercial agreement explicitly covers that scope.

## 3) Model/data artefacts and external dependencies

Code licensing and artefact licensing are not automatically identical.
An artefact-specific licence check is required before redistribution or commercial embedding.

In practice, the following must be treated separately:

- **pretrained artefacts**,
- **model weights**,
- **datasets**.

Use the model and data artefact matrix to record each external or generated source and its redistributable terms before shipping,
including provenance, dataset usage terms, and any third-party commercial constraints.
The tracked matrix lives at `security/model_data_license_matrix.json`.

Do not assume the source code licence covers every artefact in a deployment.
If any artefact has a different provenance or licence, follow that licence and keep a traceable matrix entry.

## 4) Redistribution and embedding checklist

- Verify the model and data artefact matrix is current before publication.
- Confirm source-availability obligations for any AGPL-modified service path.
- Verify licence compatibility for pretrained artefacts before redistribution.
- Verify model-weight transfer rights and dataset usage terms before commercial embedding.
- Keep audit evidence in the relevant project compliance trail.

## 5) Optional commercial licence validation

AGPL use remains the default and never requires a commercial key. Commercial
licence validation is opt-in and runs only when a caller explicitly supplies a
key through `sc_neurocore.set_license_key(...)` or sets
`SC_NEUROCORE_LICENSE_KEY`.

Install the optional HTTP dependency before validating keys:

```bash
pip install "sc-neurocore[license]"
```

Example:

```python
import sc_neurocore

status = sc_neurocore.set_license_key("...")
if status.commercial_enabled:
    print("Commercial licence active")
```

The raw key is sent to the configured validation endpoint and is not stored in
the returned process-local status object. Invalid or expired keys do not enable
commercial mode.

## 6) Practical compliance rule

Treat this boundary as enforced at build time:

1. Code uses AGPL conditions by default.
2. Commercial terms require a separate agreement where AGPL does not already allow the intended use.
3. Modified network deployments remain subject to AGPL source-availability requirements.
4. No redistribution decision is final until model, data, and weight artefacts are licence-checked.
