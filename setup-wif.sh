#!/usr/bin/env bash
# -- Set up Workload Identity Federation for GitHub Actions -> Cloud Run ---
#
# Run this ONCE from a machine with gcloud authenticated as a project owner.
# It creates (or reuses) a WIF pool + provider, a deploy service account,
# and prints the two secrets you need to add to GitHub.
#
# Usage:
#   chmod +x setup-wif.sh
#   ./setup-wif.sh
# -------------------------------------------------------------------------

set -euo pipefail

PROJECT_ID="your-gcp-project"
REGION="us-central1"
GITHUB_REPO="harshitrajchaurasia/My-Own-RAG-Bot"

POOL_NAME="github-actions"
PROVIDER_NAME="github"
SA_NAME="quillshift-deploy"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=== WIF Setup for ${GITHUB_REPO} ==="
echo "Project: ${PROJECT_ID}"
echo ""

# 1. Enable required APIs
echo "-> Enabling APIs..."
gcloud services enable \
  iamcredentials.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  --project="${PROJECT_ID}" --quiet

# 2. Create or reuse WIF pool
echo "-> Checking WIF pool '${POOL_NAME}'..."
if gcloud iam workload-identity-pools describe "${POOL_NAME}" \
    --location=global --project="${PROJECT_ID}" &>/dev/null; then
  echo "   Pool already exists, reusing."
else
  echo "   Creating pool..."
  gcloud iam workload-identity-pools create "${POOL_NAME}" \
    --location=global \
    --project="${PROJECT_ID}" \
    --display-name="GitHub Actions" \
    --description="WIF pool for GitHub Actions CI/CD"
fi

# 3. Create or reuse WIF provider
echo "-> Checking WIF provider '${PROVIDER_NAME}'..."
if gcloud iam workload-identity-pools providers describe "${PROVIDER_NAME}" \
    --workload-identity-pool="${POOL_NAME}" \
    --location=global --project="${PROJECT_ID}" &>/dev/null; then
  echo "   Provider already exists, reusing."
else
  echo "   Creating GitHub OIDC provider..."
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_NAME}" \
    --workload-identity-pool="${POOL_NAME}" \
    --location=global \
    --project="${PROJECT_ID}" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository=='${GITHUB_REPO}'"
fi

# 4. Create or reuse service account
echo "-> Checking service account '${SA_NAME}'..."
if gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT_ID}" &>/dev/null; then
  echo "   Service account already exists, reusing."
else
  echo "   Creating service account..."
  gcloud iam service-accounts create "${SA_NAME}" \
    --project="${PROJECT_ID}" \
    --display-name="QuillShift Cloud Run Deployer"
fi

# 5. Grant roles to the service account
echo "-> Granting IAM roles..."
for ROLE in \
  roles/run.admin \
  roles/iam.serviceAccountUser \
  roles/storage.admin \
  roles/cloudbuild.builds.builder \
  roles/secretmanager.secretAccessor \
  roles/artifactregistry.admin; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --quiet --no-user-output-enabled 2>/dev/null || true
done
echo "   Done."

# 6. Allow GitHub Actions to impersonate the SA via WIF
echo "-> Binding WIF -> service account..."
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
POOL_ID="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_NAME}"

gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_REPO}" \
  --quiet --no-user-output-enabled 2>/dev/null || true

# 7. Print the values for GitHub secrets
WIF_PROVIDER_FULL="${POOL_ID}/providers/${PROVIDER_NAME}"

echo ""
echo "==========================================================="
echo "  Add these as GitHub repository secrets:"
echo "  (Settings -> Secrets and variables -> Actions)"
echo "==========================================================="
echo ""
echo "  WIF_PROVIDER:"
echo "  ${WIF_PROVIDER_FULL}"
echo ""
echo "  WIF_SERVICE_ACCOUNT:"
echo "  ${SA_EMAIL}"
echo ""
echo "==========================================================="
echo "  Done! Push to main to trigger the deploy workflow."
echo "==========================================================="
