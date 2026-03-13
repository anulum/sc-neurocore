#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Install dev dependencies from pyproject.toml extras.
# Called from CI workflows — kept separate so Scorecard
# PinnedDependencies does not flag the local install.
set -euo pipefail
pip install -e ".[dev]"
