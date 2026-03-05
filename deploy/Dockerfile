# SPDX-License-Identifier: AGPL-3.0-or-later
# syntax=docker/dockerfile:1

# ── builder stage ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir --prefix=/install .

# ── runtime stage ──────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="SC-NeuroCore" \
      org.opencontainers.image.description="Universal Stochastic Computing Framework for Neuromorphic Hardware" \
      org.opencontainers.image.url="https://github.com/anulum/sc-neurocore" \
      org.opencontainers.image.source="https://github.com/anulum/sc-neurocore" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.vendor="ANULUM"

COPY --from=builder /install /usr/local

RUN useradd --create-home --shell /bin/bash scnc
USER scnc
WORKDIR /home/scnc

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import sc_neurocore" || exit 1

CMD ["python3"]
