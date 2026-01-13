# 03_CODE/sc-neurocore/deploy/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install system deps
# (e.g., build-essential if we compile extensions later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Code
COPY src/ /app/src/
COPY pyproject.toml /app/
COPY README.md /app/

# Install Deps
# We assume requirements or pyproject. Use pip directly for demo.
RUN pip install --no-cache-dir numpy scipy numba matplotlib

# Set Python Path
ENV PYTHONPATH=/app/src

# Default Command: Run Tests (if we copied them) or Python shell
CMD ["python3"]
