#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ==========================================
# VALIDATE ENVIRONMENT VARIABLES
# ==========================================
if [ -z "$DOCKER_HUB_USER" ] || [ -z "$EC2_IP" ] || [ -z "$EC2_PEM_PATH" ]; then
    echo "CRITICAL ERROR: Missing required environment variables."
    echo "Please ensure DOCKER_HUB_USER, EC2_IP, and NOT-EMPTY EC2_PEM_PATH are exported."
    exit 1
fi

# ==========================================
# SYSTEM PARAMETERS
# ==========================================
IMAGE_NAME="mlops-inference-service"
TAG="latest"
REMOTE_CONTAINER_NAME="inference-api"
REMOTE_USER="ubuntu"

REPO_TAG="${DOCKER_HUB_USER}/${IMAGE_NAME}:${TAG}"

echo "=================================================="
# Step 1: Build the Docker image from the project root context
echo "[1/6] Building Docker image..."
docker build -t "$IMAGE_NAME" -f serving/Dockerfile .

# Step 2: Tag the image with Docker Hub repository context
echo "[2/6] Tagging Docker image as ${REPO_TAG}..."
docker tag "$IMAGE_NAME" "$REPO_TAG"

# Step 3: Push the image to Docker Hub
echo "[3/6] Pushing image to Docker Hub..."
docker push "$REPO_TAG"

# Step 4: Execute remote deployment commands over SSH safely
echo "[4/6] Connecting to EC2 instance ($EC2_IP) via SSH..."
ssh -i "$EC2_PEM_PATH" -o StrictHostKeyChecking=no "${REMOTE_USER}@${EC2_IP}" << EOF
    set -e
    echo "Connected successfully to EC2 instance."

    echo "[5/6] Pulling fresh image: ${REPO_TAG}..."
    sudo docker pull "${REPO_TAG}"

    echo "[6/6] Redeploying container idempotently..."
    # Stop existing container if it exists, using sudo docker
    if [ \$(sudo docker ps -aq -f name=^/${REMOTE_CONTAINER_NAME}\$) ]; then
        echo "Stopping and removing active container: ${REMOTE_CONTAINER_NAME}"
        sudo docker stop "${REMOTE_CONTAINER_NAME}"
        sudo docker rm "${REMOTE_CONTAINER_NAME}"
    fi

    # Launch container matching port 8000 using baked-in image modules
    echo "Launching new container on port 8000..."
    sudo docker run -d \
        -p 8000:8000 \
        --name "${REMOTE_CONTAINER_NAME}" \
        --restart unless-stopped \
        "${REPO_TAG}"

    echo "Remote deployment successfully established."
EOF

echo "=================================================="
echo "SUCCESS: MLOps Inference Service deployment automation complete!"
echo "Endpoints active at http://${EC2_IP}:8000/health and http://${EC2_IP}:8000/predict"
echo "=================================================="