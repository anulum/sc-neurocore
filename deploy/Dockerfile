# SPDX-License-Identifier: AGPL-3.0-or-later
# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
ARG RUST_VERSION=stable

# ── Stage 1: Rust engine build ───────────────────────────────────
FROM python:${PYTHON_VERSION}-slim@sha256:ccc7089399c8bb65dd1fb3ed6d55efa538a3f5e7fca3f5988ac3b5b87e593bf0 AS rust-builder

ARG RUST_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential cmake pkg-config \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal  # pinned via RUST_VERSION ARG
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY requirements/maturin.txt ./requirements/maturin.txt
RUN pip install --no-cache-dir --require-hashes -r requirements/maturin.txt

# Root Cargo.toml is a workspace manifest; Cargo.lock pins deps.
# bridge/pyproject.toml references engine/Cargo.toml via manifest-path.
COPY Cargo.toml Cargo.lock ./
COPY engine/ ./engine/
COPY bridge/ ./bridge/
COPY README.md ./README.md

RUN cd bridge && maturin build --release --interpreter python3 \
    && ls /build/target/wheels/

# ── Stage 2: Python package build ────────────────────────────────
FROM python:${PYTHON_VERSION}-slim@sha256:ccc7089399c8bb65dd1fb3ed6d55efa538a3f5e7fca3f5988ac3b5b87e593bf0 AS py-builder

ARG INSTALL_EXTRAS=""

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements/build.txt ./requirements/build.txt
COPY requirements/hdl.txt ./requirements/hdl.txt
COPY requirements/hub.txt ./requirements/hub.txt
COPY requirements/runtime.txt ./requirements/runtime.txt
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir --require-hashes -r requirements/build.txt \
    && pip install --no-cache-dir --require-hashes --prefix=/install -r requirements/runtime.txt
RUN case "${INSTALL_EXTRAS}" in \
      ""|"core") true ;; \
      "hdl") pip install --no-cache-dir --require-hashes --prefix=/install -r requirements/hdl.txt ;; \
      "hub"|"studio") pip install --no-cache-dir --require-hashes --prefix=/install -r requirements/hub.txt ;; \
      *) echo "Unsupported INSTALL_EXTRAS=${INSTALL_EXTRAS}; use core, hdl, hub, or studio" >&2; exit 64 ;; \
    esac
RUN if [ "${INSTALL_EXTRAS}" = "hdl" ]; then \
        python -c "from pathlib import Path; import sys; sys.path.insert(0, 'src'); from sc_neurocore.hdl.resources import list_baseline_primitive_rtl; assert list_baseline_primitive_rtl(); assert Path('src/sc_neurocore/hdl/primitives/sc_bitstream_encoder.v').is_file()"; \
    fi
RUN python -m build --wheel --no-isolation --outdir /tmp/sc-neurocore-wheel \
    && python -m installer --prefix=/install /tmp/sc-neurocore-wheel/*.whl

COPY --from=rust-builder /build/target/wheels/*.whl /tmp/
RUN python -m installer --prefix=/install /tmp/*.whl

# ── Stage 3: Runtime ─────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim@sha256:ccc7089399c8bb65dd1fb3ed6d55efa538a3f5e7fca3f5988ac3b5b87e593bf0 AS runtime

LABEL org.opencontainers.image.title="SC-NeuroCore" \
      org.opencontainers.image.description="Universal Stochastic Computing Framework for Neuromorphic Hardware" \
      org.opencontainers.image.url="https://github.com/anulum/sc-neurocore" \
      org.opencontainers.image.source="https://github.com/anulum/sc-neurocore" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
      org.opencontainers.image.vendor="ANULUM"

RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=py-builder /install /usr/local

RUN useradd --create-home --shell /bin/bash scnc
USER scnc
WORKDIR /home/scnc

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import sc_neurocore; import sc_neurocore_engine" || exit 1

CMD ["python3"]
