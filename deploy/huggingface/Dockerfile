# GeoScan web app for a Hugging Face Docker Space (free CPU tier: 2 vCPU, 16 GB RAM).
# The Space's disk is temporary: the app restores its data from, and saves new results to,
# the private dataset named in GEOSCAN_STORE_REPO (see geoscan/webapp/persist.py).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /home/user/app
COPY --chown=user requirements-web.txt .
RUN pip install --no-cache-dir --user -r requirements-web.txt

# code only; data/ and outputs/ are restored next to it at start-up
COPY --chown=user geoscan ./geoscan
WORKDIR /home/user/app/geoscan

EXPOSE 7860
CMD ["python", "-W", "ignore", "-m", "geoscan.webapp", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
