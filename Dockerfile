# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Deterministic numerics. Without one BLAS thread and a fixed hash seed the
# runs are reproducible only up to thread scheduling.
ENV PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg RESULTS_DIR=/work/results

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran libopenblas-dev libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /work/requirements.txt

ENV PYTHONPATH=/work:/work/experiments

COPY qgloc /work/qgloc
COPY experiments /work/experiments
COPY scripts /work/scripts
COPY tests /work/tests
COPY paper /work/paper
COPY README.md Makefile setup.py /work/

RUN mkdir -p /work/results/cache && chmod +x /work/scripts/*.sh
RUN python -m pytest tests -q

# The spin-up and the initial ensembles are cached under results/cache, which
# is a mounted volume, so a second run starts in seconds.
CMD ["bash", "/work/scripts/run_all.sh"]
