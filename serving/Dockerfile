# Use an official, minimal Python runtime base image
FROM python:3.10-slim

# Set system environment execution flags
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establish the workspace directory inside the container
WORKDIR /app

# Copy requirements from the project root (the build context)
COPY requirements.txt .

# Upgrade pip and install all required project dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the serving and model directories relative to the build context
COPY serving/ ./serving/
COPY model/ ./model/

# Expose the internal port that FastAPI/Uvicorn binds to
EXPOSE 8000

# Launch the FastAPI application using Uvicorn
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]