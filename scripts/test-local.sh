#!/bin/bash

# Local GitHub Actions testing script
# This script runs GitHub Actions workflows locally using act
# with the exact same environment, versions, and configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Add act to PATH if installed in ~/.local/bin
export PATH="$PATH:$HOME/.local/bin"

echo -e "${BLUE}🚀 Starting local GitHub Actions testing...${NC}"

# Check if act is installed
if ! command -v act &> /dev/null; then
    echo -e "${RED}❌ Error: 'act' is not installed or not in PATH${NC}"
    echo "Please install act: https://github.com/nektos/act#installation"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Error: Docker is not running${NC}"
    echo "Please start Docker and try again"
    exit 1
fi

# Function to run a specific job
run_job() {
    local job_name="$1"
    local workflow_file="$2"
    
    echo -e "${YELLOW}🔧 Running job: $job_name${NC}"
    
    # Use act with our configuration
    act -W "$workflow_file" \
        --job "$job_name" \
        --env-file .env \
        --platform ubuntu-latest=catthehacker/ubuntu:act-22.04 \
        --container-architecture linux/amd64 \
        --artifact-server-path /tmp/artifacts \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Job $job_name completed successfully${NC}"
    else
        echo -e "${RED}❌ Job $job_name failed${NC}"
        return 1
    fi
}

# Function to run all jobs in a workflow
run_workflow() {
    local workflow_file="$1"
    local workflow_name=$(basename "$workflow_file" .yml)
    
    echo -e "${BLUE}🏗️  Running workflow: $workflow_name${NC}"
    
    act -W "$workflow_file" \
        --env-file .env \
        --platform ubuntu-latest=catthehacker/ubuntu:act-22.04 \
        --container-architecture linux/amd64 \
        --artifact-server-path /tmp/artifacts \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Workflow $workflow_name completed successfully${NC}"
    else
        echo -e "${RED}❌ Workflow $workflow_name failed${NC}"
        return 1
    fi
}

# Main execution
case "${1:-all}" in
    "lint")
        run_job "lint" ".github/workflows/build.yml"
        ;;
    "test")
        run_job "test" ".github/workflows/build.yml"
        ;;
    "security")
        run_job "security" ".github/workflows/build.yml"
        ;;
    "notebooks")
        run_job "notebooks" ".github/workflows/build.yml"
        ;;
    "build")
        run_workflow ".github/workflows/build.yml"
        ;;
    "release")
        echo -e "${YELLOW}⚠️  Release workflow requires secrets and should be run with caution${NC}"
        read -p "Are you sure you want to run the release workflow locally? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_workflow ".github/workflows/release.yml"
        else
            echo "Cancelled."
            exit 0
        fi
        ;;
    "all"|"")
        echo -e "${BLUE}🔄 Running all build workflow jobs...${NC}"
        run_workflow ".github/workflows/build.yml"
        ;;
    "list")
        echo -e "${BLUE}📋 Available commands:${NC}"
        echo "  lint      - Run linting job only"
        echo "  test      - Run test job only"
        echo "  security  - Run security job only"
        echo "  notebooks - Run notebooks job only"
        echo "  build     - Run complete build workflow"
        echo "  release   - Run release workflow (use with caution)"
        echo "  all       - Run complete build workflow (default)"
        echo "  list      - Show this help"
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo "Use '$0 list' to see available commands"
        exit 1
        ;;
esac

echo -e "${GREEN}🎉 Local testing completed!${NC}"