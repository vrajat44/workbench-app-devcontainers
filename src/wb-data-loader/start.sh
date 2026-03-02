#!/bin/bash
# WB_Data_Loader (PyAirbyte) — Startup Script

echo "🚀 Starting WB Data Loader (PyAirbyte edition)..."

# GCP project for BigQuery destination
if [ -z "$GCP_PROJECT_ID" ]; then
    GCP_PROJECT_ID=$(curl -s -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/project/project-id" 2>/dev/null || echo "")
fi

if [ -z "$GCP_PROJECT_ID" ]; then
    echo "⚠ WARNING: No GCP_PROJECT_ID detected. BigQuery sync may not work."
fi

CMD="python app.py --port=8080"

if [ -n "$GCP_PROJECT_ID" ]; then
    CMD="${CMD} --project=${GCP_PROJECT_ID}"
fi

if [ -n "$BQ_DATASET" ]; then
    CMD="${CMD} --bq-dataset=${BQ_DATASET}"
fi

if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    CMD="${CMD} --credentials-path=${GOOGLE_APPLICATION_CREDENTIALS}"
fi

echo "   Project:     ${GCP_PROJECT_ID:-<not set>}"
echo "   BQ Dataset:  ${BQ_DATASET:-<not set>}"
echo "   Credentials: ${GOOGLE_APPLICATION_CREDENTIALS:-ADC}"
echo "   Port:        8080"
echo ""

exec ${CMD}
