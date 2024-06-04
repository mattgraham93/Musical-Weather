FROM python:3.10-slim

ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
ENV PORT 5000

WORKDIR $APP_HOME

# Copy everything from the current directory into /app in the Docker image
COPY . ./

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app