# SPDX-License-Identifier: AGPL-3.0-or-later
"""Install engine test dependencies from pyproject.toml extras."""

import subprocess
import sys

raise SystemExit(subprocess.call([sys.executable, "-m", "pip", "install", "-e", ".[dev-full]"]))
