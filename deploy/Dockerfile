# 03_CODE/sc-neurocore/deploy/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install system deps
# (e.g., build-essential if we compile extensions later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE /app/
COPY src/ /app/src/

# Install the package properly via pyproject.toml
RUN pip install --no-cache-dir .

# Verify installation
RUN python -c "import sc_neurocore; print(f'sc-neurocore {sc_neurocore.__version__} installed')"

# Default Command
CMD ["python3"]
