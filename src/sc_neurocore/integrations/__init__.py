# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Framework integrations (Lava/Loihi, etc.)

"""Framework integrations (Lava/Loihi, etc.).

Always importable:
- ``HAS_LAVA``           — bool flag, True when ``lava-nc`` is on the path
- ``LoihiNetworkConfig`` — dataclass for compiled Loihi layer config
- ``SCtoLavaConverter``  — SC dense layer → ``LoihiNetworkConfig``
- ``export_weights_loihi`` — SC ``[0, 1]`` weights → signed Q8 ints
- ``loihi_threshold_from_sc`` — SC normalised threshold → integer

Importable only when ``HAS_LAVA`` is True (i.e. when ``lava-nc`` is
installed; the package supports Python 3.10):
- ``SCDenseProcess``  — Lava ``Process`` wrapping an SC dense layer
- ``PySCDenseModel``  — Loihi-protocol Python model implementation

Direct imports of ``SCDenseProcess`` / ``PySCDenseModel`` from this
package raise ``ImportError`` when ``lava-nc`` is missing — callers
should branch on ``HAS_LAVA`` first.
"""

from .lava_bridge import (
    HAS_LAVA,
    LoihiNetworkConfig,
    SCtoLavaConverter,
    export_weights_loihi,
    loihi_threshold_from_sc,
)

__all__ = [
    "HAS_LAVA",
    "LoihiNetworkConfig",
    "SCtoLavaConverter",
    "export_weights_loihi",
    "loihi_threshold_from_sc",
]

if HAS_LAVA:
    from .lava_bridge import PySCDenseModel, SCDenseProcess

    __all__ += ["SCDenseProcess", "PySCDenseModel"]
