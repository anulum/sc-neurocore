# SPDX-License-Identifier: AGPL-3.0-or-later
"""Install dev dependencies from pyproject.toml extras."""

import subprocess
import sys

raise SystemExit(subprocess.call([sys.executable, "-m", "pip", "install", "-e", ".[dev]"]))
