#!/usr/bin/env bash
# -- Deploy QuillShift to Google Cloud Run ---------------------------------
# Prerequisites:
#   1. gcloud CLI installed and authenticated: gcloud auth login
#   2. A GCP project: gcloud config set project YOUR_PROJECT_ID
#   3. Your GOOGLE_API_KEY ready
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
# ------------------------------------------------------------------------

set -euo pipefail

# -- Configuration ---------------------------------------------------------
SERVICE_NAME="quillshift"
REGION="${GCP_REGION:-us-central1}"
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
  echo "Error: No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

echo "+==================================================+"
echo "|           QuillShift -> Cloud Run Deploy            |"
echo "+==================================================+"
echo "|  Project:  $PROJECT_ID"
echo "|  Service:  $SERVICE_NAME"
echo "|  Region:   $REGION"
echo "+==================================================+"
echo ""

# -- Step 1: Enable required APIs -----------------------------------------
echo "-> Enabling Cloud Run & Artifact Registry APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com --quiet

# -- Step 2: Set the API key as a secret (if not already) --------------
if ! gcloud secrets describe quillshift-google-api-key --project="$PROJECT_ID" &>/dev/null; then
  echo ""
  echo "-> Creating secret for GOOGLE_API_KEY..."
  echo "  Paste your Google API key and press Enter:"
  read -rs GOOGLE_API_KEY
  echo "$GOOGLE_API_KEY" | gcloud secrets create quillshift-google-api-key \
    --data-file=- --project="$PROJECT_ID" --quiet
  echo "  Secret created."
else
  echo "-> Secret 'quillshift-google-api-key' already exists."
fi

# Grant Cloud Run access to the secret
echo "-> Granting Cloud Run access to the secret..."
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding quillshift-google-api-key \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project="$PROJECT_ID" --quiet 2>/dev/null || true

# -- Step 3: Build and deploy --------------------------------------------
echo ""
echo "-> Building and deploying to Cloud Run..."
echo "  (This takes 2-4 minutes on first deploy)"
echo ""

gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 10 \
  --timeout 300 \
  --session-affinity \
  --set-secrets="GOOGLE_API_KEY=quillshift-google-api-key:latest" \
  --set-env-vars="LLM_PROVIDER=gemini,SESSION_RPM=10,GLOBAL_RPD=200,SESSION_TTL_HOURS=2" \
  --quiet

# -- Done ------------------------------------------------------------------
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format="value(status.url)")

echo ""
echo "+==================================================+"
echo "|             Deploy Successful!                   |"
echo "+==================================================+"
echo "|  URL: $SERVICE_URL"
echo "|                                                  |"
echo "|  Share this link with friends & recruiters!      |"
echo "|  Costs: \$0 (scales to zero when idle)            |"
echo "+==================================================+"
