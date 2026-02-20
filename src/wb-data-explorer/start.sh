#!/bin/bash
# WB Data Explorer — Startup Script
# Auto-detects GCP project from Workbench environment or env vars.
# Supports metadata from local dir or GCS URI (gs://bucket/prefix).

set -e

# Metadata source: local path or GCS URI
# Set METADATA_SOURCE env var to a gs:// URI to load from GCS at runtime
METADATA_SOURCE="${METADATA_SOURCE:-${METADATA_JSON_DIR:-/app/metadata}}"

# Auto-detect project IDs from environment or gcloud
if [ -z "$GCP_PROJECT_ID" ]; then
    GCP_PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
fi

if [ -z "$GCP_PROJECT_ID" ]; then
    echo "⚠ WARNING: No GCP_PROJECT_ID set. Running in metadata-only mode."
    echo "  Set GCP_PROJECT_ID environment variable or run 'gcloud config set project <id>'"
fi

# Build the command
CMD="python app.py --port=8080 --json-dir=${METADATA_SOURCE}"

if [ -n "$GCP_PROJECT_ID" ]; then
    CMD="${CMD} --project=${GCP_PROJECT_ID}"
fi

# Data projects (space-separated list in env var)
if [ -n "$DATA_PROJECT_IDS" ]; then
    for proj in $DATA_PROJECT_IDS; do
        CMD="${CMD} --data-project ${proj}"
    done
elif [ -n "$DATA_PROJECT_ID" ]; then
    CMD="${CMD} --data-project ${DATA_PROJECT_ID}"
fi

# LLM model override
if [ -n "$LLM_MODEL" ]; then
    CMD="${CMD} --llm-model=${LLM_MODEL}"
fi

echo "🚀 Starting WB Data Explorer..."
echo "   Project: ${GCP_PROJECT_ID:-<not set>}"
echo "   Data Projects: ${DATA_PROJECT_IDS:-${DATA_PROJECT_ID:-<same as project>}}"
echo "   Metadata: ${METADATA_SOURCE}"
echo "   Port: 8080"
echo ""

exec $CMD
