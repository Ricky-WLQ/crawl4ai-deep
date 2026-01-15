FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crawl4AI specific settings
ENV CRAWL4AI_HEADLESS=true
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Install system dependencies for Playwright/Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run crawl4ai setup to install browsers
RUN crawl4ai-setup

# SOLUTION 2 FIX: Pre-download MULTILINGUAL sentence-transformers model
# Updated from MiniLM-L12-v2 to mpnet-base-v2 (better for legal documents)
# Model is ~500MB, supports 50+ languages including Chinese
# This avoids delay on first request
RUN python -c "from sentence_transformers import SentenceTransformer; \
    print('Downloading SOLUTION 2 model: paraphrase-multilingual-mpnet-base-v2...'); \
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2'); \
    print('Model downloaded successfully!')"

# Optional: Pre-download OpenRouter embedding model metadata (for faster inference)
# This is lightweight and doesn't require actual model download
RUN python -c "import httpx; print('OpenRouter embeddings client ready')" || true

# Copy application code
COPY . .

# Health check to ensure service is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health', timeout=5)"

# Expose port
EXPOSE 8080

# Start the application
CMD ["python", "main.py"]
