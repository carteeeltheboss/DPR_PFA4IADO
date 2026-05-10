# ───────────────────────────────────────────────────────────────
# MedFusionNet – Docker image for the Flask web service
# Build:  docker build -t medfusionnet .
# Run:    docker run -p 8000:8000 medfusionnet
# ───────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # matplotlib needs a writable config dir
    MPLCONFIGDIR=/tmp/matplotlib \
    # Force CPU device inside the container (no GPU passthrough by default)
    CUDA_VISIBLE_DEVICES="" \
    # Use non-interactive matplotlib backend
    MPLBACKEND=Agg

WORKDIR /app

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────
# Install CPU-only PyTorch first (much smaller than the full CUDA build)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy only the requirements files first for better layer caching
COPY DPR_MedFusionNet/requirements.txt /tmp/req-med.txt
COPY DPR_WebService/requirements.txt   /tmp/req-web.txt

# Install remaining dependencies (timm, Pillow, matplotlib, Flask, scikit-learn)
# The web requirements.txt references the med requirements via relative path,
# so we install them explicitly to avoid path issues inside Docker.
RUN pip install --no-cache-dir \
    timm>=1.0.0 \
    "numpy>=1.26.0" \
    "Pillow>=10.0.0" \
    "matplotlib>=3.8.0" \
    "Flask>=3.0.0" \
    "scikit-learn>=1.3.0"

# ── Application code ─────────────────────────────────────────
# Copy the full project (respects .dockerignore)
COPY . .

# ── Runtime ───────────────────────────────────────────────────
# Create directories the app expects at runtime
RUN mkdir -p DPR_WebService/runtime /tmp/matplotlib

# The Flask app binds to 127.0.0.1 by default — override to 0.0.0.0
# so Docker port-forwarding works.
EXPOSE 8000

CMD ["python", "DPR_WebService/app.py"]
