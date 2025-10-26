# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for production
ENV PYTHONUNBUFFERED=True
ENV APP_HOME=/app
ENV PORT=8080
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Set the working directory in the container
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . ./

# Download NLTK data during image build (with better error handling)
RUN python -c "import nltk; \
    nltk.download('vader_lexicon', quiet=True); \
    nltk.download('stopwords', quiet=True); \
    nltk.download('punkt', quiet=True); \
    nltk.download('wordnet', quiet=True); \
    print('NLTK data downloaded successfully')" || \
    echo "Warning: NLTK download failed, app will run in limited mode"

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app $APP_HOME
USER app

# Expose the port
EXPOSE $PORT

# Run the application with gunicorn
CMD exec gunicorn \
    --bind :$PORT \
    --workers 1 \
    --threads 2 \
    --timeout 60 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    cloud_main:app