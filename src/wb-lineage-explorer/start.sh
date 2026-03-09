#!/bin/bash
# WB Lineage Explorer — Startup Script
# Reads config from environment variables (set in docker-compose.yaml or Workbench)

echo "🚀 Starting WB Lineage Explorer..."

# GCP project for billing (BQ jobs)
# In Workbench, this is auto-set or can be detected from metadata server
if [ -z "$GCP_PROJECT_ID" ]; then
    # Try GCE metadata server (available in Workbench VMs)
    GCP_PROJECT_ID=$(curl -s -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/project/project-id" 2>/dev/null || echo "")
fi

if [ -z "$GCP_PROJECT_ID" ]; then
    echo "⚠ WARNING: No GCP_PROJECT_ID detected. BigQuery features may not work."
fi

echo "   Project: ${GCP_PROJECT_ID:-<not set>}"
echo "   Port:    8080"
echo ""

exec python app.py --port=8080
