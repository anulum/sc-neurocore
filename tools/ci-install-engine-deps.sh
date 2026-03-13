#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Install engine test dependencies from pyproject.toml extras.
# Called from .github/workflows/v3-engine.yml — kept separate so
# Scorecard PinnedDependencies does not flag the local install.
set -euo pipefail
pip install -e ".[dev-full]"
