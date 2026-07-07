#!/usr/bin/env bash
# upload_data.sh — Upload raw data to a Databricks Unity Catalog Volume
#
# Usage:
#   ./scripts/upload_data.sh                          # uses defaults
#   CATALOG=dev_quickstart_catalog ./scripts/upload_data.sh
#   ./scripts/upload_data.sh --catalog dev_quickstart_catalog --schema healthcare_ml_dev
#
set -euo pipefail

# ── Defaults (override via env vars or flags) ─────────────────────────────────
CATALOG="${CATALOG:-healthcare_catalog}"
SCHEMA="${SCHEMA:-healthcare_ml}"
VOLUME="raw"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --catalog) CATALOG="$2"; shift 2 ;;
    --schema)  SCHEMA="$2";  shift 2 ;;
    --volume)  VOLUME="$2";  shift 2 ;;
    --data-dir) LOCAL_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

VOLUME_PATH="/Volumes/${CATALOG}/${SCHEMA}/${VOLUME}"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if ! command -v databricks &>/dev/null; then
  echo "ERROR: Databricks CLI not found. Install it with: pip install databricks-cli" >&2
  exit 1
fi

# Auth: prefer explicit env vars; fall back to CLI profile (~/.databrickscfg) or OAuth
if [[ -n "${DATABRICKS_HOST:-}" && -n "${DATABRICKS_TOKEN:-}" ]]; then
  echo "  Auth   : env vars (DATABRICKS_HOST / DATABRICKS_TOKEN)"
elif databricks auth env --host "${DATABRICKS_HOST:-}" &>/dev/null 2>&1; then
  echo "  Auth   : Databricks CLI profile"
else
  echo "  Auth   : using active Databricks CLI session"
fi

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "ERROR: Data directory not found: $LOCAL_DIR" >&2
  exit 1
fi

# ── Upload ─────────────────────────────────────────────────────────────────────
echo "Uploading files from ${LOCAL_DIR} → ${VOLUME_PATH}"
echo "  Catalog: ${CATALOG}  Schema: ${SCHEMA}  Volume: ${VOLUME}"
echo

shopt -s nullglob
files=("$LOCAL_DIR"/*)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No files found in ${LOCAL_DIR}. Nothing to upload."
  exit 0
fi

for file in "${files[@]}"; do
  [[ -f "$file" ]] || continue
  filename="$(basename "$file")"
  dest="${VOLUME_PATH}/${filename}"
  echo "  → ${filename}"
  databricks fs cp --overwrite "$file" "dbfs:${dest}" 2>/dev/null || \
    databricks fs cp --overwrite "$file" "${dest}"
done

echo
echo "Upload complete."
