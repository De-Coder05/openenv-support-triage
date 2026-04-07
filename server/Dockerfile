FROM python:3.10-slim

WORKDIR /app

# Install dependencies and clean up apt cache (2 vCPU, 8GB constraint)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install -e .

# Copy application files
COPY . .

# Run the OpenEnv server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
