# SPDX-License-Identifier: AGPL-3.0-or-later
# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
ARG RUST_VERSION=stable

# ── Stage 1: Rust engine build ───────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS rust-builder

ARG RUST_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir maturin>=1.5

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY engine/ ./engine/
COPY bridge/ ./bridge/
COPY README.md ./README.md

RUN cd bridge && maturin build --release --interpreter python3 \
    && ls /build/target/wheels/

# ── Stage 2: Python package build ────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS py-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir --prefix=/install .

COPY --from=rust-builder /build/target/wheels/*.whl /tmp/
RUN pip install --no-cache-dir --prefix=/install /tmp/*.whl

# ── Stage 3: Runtime ─────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="SC-NeuroCore" \
      org.opencontainers.image.description="Universal Stochastic Computing Framework for Neuromorphic Hardware" \
      org.opencontainers.image.url="https://github.com/anulum/sc-neurocore" \
      org.opencontainers.image.source="https://github.com/anulum/sc-neurocore" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.vendor="ANULUM"

COPY --from=py-builder /install /usr/local

RUN useradd --create-home --shell /bin/bash scnc
USER scnc
WORKDIR /home/scnc

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import sc_neurocore; import sc_neurocore_engine" || exit 1

CMD ["python3"]
