FROM python:3.11-slim

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency spec first (layer cache)
COPY pyproject.toml ./
COPY src/ ./src/

# Install package (CPU-only torch to keep image smaller)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .

# Copy application code
COPY app/ ./app/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Data directories (mount at runtime — do not bake data into image)
RUN mkdir -p data/raw data/processed data/synthetic outputs/models outputs/reports

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
