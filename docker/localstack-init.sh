#!/bin/bash
# All variables are injected from .env via docker-compose at runtime.
set -e

_upsert_secret() {
  local name="$1" value="$2"
  awslocal secretsmanager create-secret \
    --name "${name}" --secret-string "${value}" 2>/dev/null || \
  awslocal secretsmanager put-secret-value \
    --secret-id "${name}" --secret-string "${value}"
}

_upsert_secret "seshat/postgres_url" "${DATABASE_URL}"
_upsert_secret "seshat/root-api-key" "${SESHAT_ROOT_API_KEY}"

_upsert_secret "seshat/openai_api_key"       "${OPENAI_API_KEY}"
_upsert_secret "seshat/anthropic_api_key"    "${ANTHROPIC_API_KEY}"
_upsert_secret "seshat/azure_openai_api_key" "${AZURE_OPENAI_API_KEY}"
_upsert_secret "seshat/assemblyai_api_key"   "${ASSEMBLYAI_API_KEY}"
_upsert_secret "seshat/cohere_api_key"      "${COHERE_API_KEY}"
_upsert_secret "seshat/voyage_api_key"      "${VOYAGE_API_KEY}"

awslocal s3 mb "s3://${S3_BUCKET}" 2>/dev/null || true

echo "LocalStack init complete"
