# WB Data Loader (PyAirbyte) — Gradio App for Verily Workbench
# Uses Python 3.12 for PyAirbyte compatibility (numpy/pyarrow constraints)
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./

# Copy startup script
COPY start.sh ./
RUN chmod +x start.sh

# Gradio default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Start the app
CMD ["./start.sh"]
