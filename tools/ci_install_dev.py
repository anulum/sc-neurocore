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
"""

from ci_install_common import install_editable


raise SystemExit(install_editable("dev,nir,compression,training,research,bioware,studio"))
