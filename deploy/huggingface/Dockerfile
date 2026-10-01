# Hugging Face Space for the public demo (ADR-020). Builds from a tagged
# release on GitHub, so the Space runs exactly the code that was tagged.
FROM python:3.12-slim

ARG REF=v1.0.0
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user

WORKDIR /app
RUN git clone --depth 1 --branch "${REF}" \
        https://github.com/Pranay777777/evidence-grounded-resume-engine.git . \
    && pip install --no-cache-dir ".[embeddings]"

ENV DEMO=true \
    AUTH=off \
    HOST=0.0.0.0 \
    PORT=7860 \
    EMBEDDER=bge-small \
    DATABASE_URL=sqlite:////tmp/demo.db \
    MODEL_CACHE_DIR=/app/.cache/models \
    SEMANTIC_CACHE_PATH=/tmp/semantic-cache.jsonl \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/app/.cache/huggingface

# Download the embedding and NLI models into the image: no cold-start download.
RUN python -m grounded.demo --warm && chown -R user:user /app
USER user
EXPOSE 7860
CMD ["python", "-m", "grounded.demo"]
