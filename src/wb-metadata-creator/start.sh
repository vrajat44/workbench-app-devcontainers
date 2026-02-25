#!/bin/bash
# WB Metadata Creator — Startup Script
# Reads config from environment variables

echo "🚀 Starting WB Metadata Creator..."

# Existing metadata source: GCS URI or local path (for cross-reference)
METADATA_SOURCE="${METADATA_SOURCE:-}"

# Output GCS bucket for saving generated metadata
OUTPUT_GCS_BUCKET="${OUTPUT_GCS_BUCKET:-}"

# GCP project for billing (BQ jobs, Vertex AI)
if [ -z "$GCP_PROJECT_ID" ]; then
    # Try GCE metadata server (available in Workbench VMs)
    GCP_PROJECT_ID=$(curl -s -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/project/project-id" 2>/dev/null || echo "")
fi

if [ -z "$GCP_PROJECT_ID" ]; then
    echo "⚠ WARNING: No GCP_PROJECT_ID detected. LLM and BQ features will not work."
fi

# Build the command
CMD="python app.py --port=8080"

if [ -n "$GCP_PROJECT_ID" ]; then
    CMD="${CMD} --project=${GCP_PROJECT_ID}"
fi

# Data projects (space-separated list in env var)
if [ -n "$DATA_PROJECT_IDS" ]; then
    CMD="${CMD} --data-project ${DATA_PROJECT_IDS}"
fi

# Existing metadata path (for cross-reference)
if [ -n "$METADATA_SOURCE" ]; then
    CMD="${CMD} --json-dir=${METADATA_SOURCE}"
fi

# Output GCS bucket
if [ -n "$OUTPUT_GCS_BUCKET" ]; then
    CMD="${CMD} --output-bucket=${OUTPUT_GCS_BUCKET}"
fi

echo "   Project:        ${GCP_PROJECT_ID:-<not set>}"
echo "   Data Projects:  ${DATA_PROJECT_IDS:-<not set>}"
echo "   Metadata Ref:   ${METADATA_SOURCE:-<not set>}"
echo "   Output Bucket:  ${OUTPUT_GCS_BUCKET:-<not set>}"
echo "   Port:           8080"
echo ""
echo "Running: ${CMD}"
echo ""

exec ${CMD}
