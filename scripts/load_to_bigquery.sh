#!/usr/bin/env bash
#
# Load the generated CSVs into a BigQuery "raw" dataset.
# Requires the gcloud/bq CLI, authenticated to your project.
#
# Usage:
#   bash scripts/load_to_bigquery.sh YOUR_GCP_PROJECT_ID [LOCATION]
#   (LOCATION defaults to EU)

set -euo pipefail

PROJECT="${1:?usage: load_to_bigquery.sh PROJECT_ID [LOCATION]}"
LOCATION="${2:-EU}"
RAW_DATASET="neobank_raw"

echo "Creating dataset ${PROJECT}:${RAW_DATASET} (if it does not exist)..."
bq --location="${LOCATION}" mk -f --dataset "${PROJECT}:${RAW_DATASET}" || true

echo "Loading raw_users..."
bq --location="${LOCATION}" load --autodetect --replace --source_format=CSV \
  "${PROJECT}:${RAW_DATASET}.raw_users" data/raw_users.csv

echo "Loading raw_events..."
bq --location="${LOCATION}" load --autodetect --replace --source_format=CSV \
  "${PROJECT}:${RAW_DATASET}.raw_events" data/raw_events.csv

echo "Done. Loaded raw_users and raw_events into ${PROJECT}:${RAW_DATASET}."
