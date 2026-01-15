FROM python:3.11-slim

LABEL "language"="python"
LABEL "framework"="fastapi"
LABEL "version"="3.7.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CRAWL4AI_HEADLESS=true
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV HF_HOME=/app/.cache/huggingface

# Install system dependencies for Playwright/Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    libu2f-udev \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --compile -r requirements.txt && \
    pip install --no-cache-dir playwright

# Install Playwright browsers (Chromium only - saves ~400MB)
RUN python -m playwright install chromium && \
    python -m playwright install-deps chromium && \
    rm -rf /tmp/* /var/tmp/*

# SOLUTION 2: Pre-download multilingual embedding model
# paraphrase-multilingual-mpnet-base-v2 (~500MB)
# Supports 50+ languages including Chinese, English, Portuguese
RUN python << 'EOF'
import sys
try:
    print("Downloading SOLUTION 2 model: paraphrase-multilingual-mpnet-base-v2...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    print("✓ Model downloaded successfully! Size: ~500MB", flush=True)
    print("✓ Supports 50+ languages including Chinese, English, Portuguese", flush=True)
except Exception as e:
    print(f"❌ Error downloading model: {e}", flush=True)
    sys.exit(1)
EOF

# Clean up cache and temporary files to reduce image size
RUN rm -rf /tmp/* /var/tmp/* /root/.cache/* && \
    find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copy application code (AFTER dependencies are installed)
COPY main.py .

# FIX: Copy .env.example using RUN with proper error handling
# This is optional, so we use shell commands
RUN if [ -f .env.example ]; then cp .env.example .; fi || true

# Expose port
EXPOSE 8080

# ✅ FIX: Health check with SIMPLE single-line command (no heredoc)
# Multi-line python << 'EOF' syntax doesn't work in HEALTHCHECK CMD
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; response = httpx.get('http://localhost:8080/health', timeout=5); exit(0 if response.status_code == 200 else 1)" || exit 1

# Start the application
CMD ["python", "main.py"]
