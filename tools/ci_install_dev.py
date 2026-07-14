# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Install dev dependencies from pyproject.toml extras

"""Install dev dependencies from pyproject.toml extras.

``training`` pulls torch (needed by arcane_zenith, darts_sc_nas, and ~300
torch-gated tests), ``research`` adds matplotlib + networkx for viz
modules, ``bioware`` adds scikit-learn, ``studio`` adds fastapi + uvicorn
for the web UI surface. Without these, ``pytest.importorskip`` skips the
tests and coverage drops below the 99 % gate even though the code is
reachable.

``federation`` (scpn-studio-platform) is added only on Python >= 3.12, the
platform SDK's floor: it makes the studio-federation conformance tests under
``tests/test_federation`` run (instead of ``pytest.importorskip``-skipping), so a
schema-A manifest or evidence-bundle drift reds the (required) 3.12 test job.
"""

import sys

from ci_install_common import install_editable

_EXTRAS = "dev,units,nir,compression,training,research,bioware,studio,julia"
if sys.version_info >= (3, 12):
    _EXTRAS += ",federation"

raise SystemExit(install_editable(_EXTRAS))
