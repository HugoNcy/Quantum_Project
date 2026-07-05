# Reproducible environment for the photonic generative models project.
# Built and run with Podman (works identically with Docker).
FROM docker.io/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Set to any non-empty value (e.g. "1") ONLY if your network intercepts TLS and
# the build fails with CERTIFICATE_VERIFY_FAILED (see README troubleshooting).
# It skips certificate verification for the package indexes below; leave empty
# for normal verified downloads.
ARG PIP_TRUST_INDEX_HOSTS=""

# CPU-only torch wheel: ~10x smaller than the default CUDA wheel.
# Exact simulation in this project is CPU-bound; no GPU needed.
RUN pip install ${PIP_TRUST_INDEX_HOSTS:+--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org} \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install ${PIP_TRUST_INDEX_HOSTS:+--trusted-host pypi.org --trusted-host files.pythonhosted.org} \
    -r requirements.txt

# Source is volume-mounted in development (see compose.yaml);
# copying it here makes the image usable standalone too.
COPY . .

EXPOSE 8888

CMD ["bash"]
