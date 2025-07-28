#!/bin/bash

# Setup script for local GitHub Actions testing
# This script installs and configures act for local CI testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Setting up local GitHub Actions testing environment...${NC}"

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${YELLOW}⚠️  Docker is not running${NC}"
    echo "Please start Docker and try again"
    echo "On most systems: sudo systemctl start docker"
    exit 1
fi

# Install act if not already installed
if ! command -v act &> /dev/null; then
    echo -e "${YELLOW}📦 Installing act...${NC}"
    mkdir -p ~/.local/bin
    curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | bash -s -- -b ~/.local/bin
    
    # Add to PATH for current session
    export PATH="$PATH:$HOME/.local/bin"
    
    # Add to shell profile
    if [[ "$SHELL" == *"bash"* ]]; then
        echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
    elif [[ "$SHELL" == *"zsh"* ]]; then
        echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.zshrc
    fi
    
    echo -e "${GREEN}✅ act installed successfully${NC}"
else
    echo -e "${GREEN}✅ act is already installed${NC}"
fi

# Verify act installation
act --version

# Pull the required Docker images for faster execution
echo -e "${YELLOW}📥 Pulling required Docker images...${NC}"
docker pull catthehacker/ubuntu:act-22.04

echo -e "${GREEN}🎉 Local testing environment setup completed!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Run: ./scripts/test-local.sh list  # See available commands"
echo "2. Run: ./scripts/test-local.sh build # Test complete workflow"
echo "3. Run individual jobs as needed"
echo ""
echo -e "${YELLOW}Note: First run may take longer as it downloads base images${NC}"