# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native Rust C-FFI bridges

"""Re-export native Rust bridge availability flags."""

from .core_engine_bridge import is_available as core_engine_available
from .learning_bridge import is_available as learning_available

__all__ = ["core_engine_available", "learning_available"]
