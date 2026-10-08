# Use official Python lightweight image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables to prevent Python from writing .pyc files
# and to ensure stdout is not buffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (needed for compiling some python packages like chroma/hnswlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir streamlit

# Copy the rest of the application
COPY . .

# Expose ports for both FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# We will use docker-compose to specify the exact entrypoint command (api vs ui)
CMD ["python", "app.py"]
