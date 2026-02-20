#!/bin/bash
# WB Data Explorer — Startup Script
# Reads config from environment variables (set in docker-compose.yaml)

echo "🚀 Starting WB Data Explorer..."

# Metadata source: GCS URI or local path
METADATA_SOURCE="${METADATA_SOURCE:-gs://metadata-json-wb-shrewd-papaya-8403}"

# GCP project for billing (BQ jobs, Vertex AI)
# In Workbench, this is auto-set or can be detected from metadata server
if [ -z "$GCP_PROJECT_ID" ]; then
    # Try GCE metadata server (available in Workbench VMs)
    GCP_PROJECT_ID=$(curl -s -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/project/project-id" 2>/dev/null || echo "")
fi

if [ -z "$GCP_PROJECT_ID" ]; then
    echo "⚠ WARNING: No GCP_PROJECT_ID detected. LLM and BQ features may not work."
fi

# Build the command
CMD="python app.py --port=8080 --server-name=0.0.0.0 --json-dir=${METADATA_SOURCE}"

if [ -n "$GCP_PROJECT_ID" ]; then
    CMD="${CMD} --project=${GCP_PROJECT_ID}"
fi

# Data projects (space-separated list in env var)
if [ -n "$DATA_PROJECT_IDS" ]; then
    for proj in $DATA_PROJECT_IDS; do
        CMD="${CMD} --data-project ${proj}"
    done
fi

# LLM model override
if [ -n "$LLM_MODEL" ]; then
    CMD="${CMD} --llm-model=${LLM_MODEL}"
fi

echo "   Project:      ${GCP_PROJECT_ID:-<not set>}"
echo "   Data Projects: ${DATA_PROJECT_IDS:-<not set>}"
echo "   Metadata:     ${METADATA_SOURCE}"
echo "   Model:        ${LLM_MODEL:-gemini-2.5-pro}"
echo "   Port:         8080"
echo ""
echo "Running: ${CMD}"
echo ""

exec ${CMD}
