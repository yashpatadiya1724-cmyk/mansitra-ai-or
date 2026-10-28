#!/bin/bash
# AWS EC2 Deployment Script for Manasitra Backend
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# This script contains the commands needed to run on your AWS EC2 instance
# to set up the environment, clone the backend code, and deploy the services.

echo "=========================================================="
echo "Starting Manasitra AWS EC2 Server Setup"
echo "=========================================================="

# 1. Update and install prerequisites
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y curl git ufw

# 2. Install Docker
echo "Installing Docker..."
sudo apt-get install -y apt-transport-https ca-certificates gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# 3. Install Docker Compose
echo "Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 4. Add current user to Docker group
sudo usermod -aG docker $USER

# 5. Configure Firewall (UFW)
echo "Configuring firewall..."
sudo ufw allow 22/tcp
sudo ufw allow 8000/tcp
sudo ufw allow 11434/tcp
sudo ufw --force enable

echo "=========================================================="
echo "Prerequisites installed successfully!"
echo "Next Steps to deploy:"
echo "1. Clone/Transfer your Manasitra project folder to this EC2 instance."
echo "2. Navigate to 'manasitra/backend/'"
echo "3. Run: docker-compose up --build -d"
echo "4. (Optional) Run: docker exec -it backend-ollama-1 ollama pull llama3"
echo "=========================================================="
