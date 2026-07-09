FROM python:3.11-slim

# ffmpeg CLI only -- no OpenCV/PIL, keeps the image small and dependencies minimal.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/

# --- Submission-only key baking -----------------------------------------
# The eval harness runs this image with no injected credentials, so for a
# public submission image the Fireworks key(s) must be baked in at build
# time via --build-arg. See README.md for the PROMINENT SECURITY WARNING
# about using disposable, spend-capped keys for this.
ARG FIREWORKS_API_KEY=""
ARG VISION_MODEL=""
ARG TEXT_MODEL=""
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY}
ENV VISION_MODEL=${VISION_MODEL}
ENV TEXT_MODEL=${TEXT_MODEL}
# --------------------------------------------------------------------------

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "agent.main"]
