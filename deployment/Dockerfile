# Multi-stage build for EV Charging LLM API
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Copy application code
COPY deployment/ ./deployment/
COPY evaluation/ ./evaluation/
COPY scripts/ ./scripts/
COPY config.yaml ./

# Copy model artifacts (these will be mounted or copied during CI/CD)
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "deployment.serve_model:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy all source code for development
COPY . .

# Create directories and set permissions
RUN mkdir -p /app/logs /app/tmp && \
    chown -R appuser:appuser /app

USER appuser

# Development server with hot reload
CMD ["python", "-m", "uvicorn", "deployment.serve_model:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 