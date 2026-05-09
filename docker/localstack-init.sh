#!/bin/bash
# All variables are injected from .env via docker-compose at runtime.
set -e

POSTGRES_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}"

awslocal secretsmanager create-secret \
  --name "${POSTGRES_SECRET_NAME}" \
  --secret-string "${POSTGRES_URL}" \
  --region "${DEFAULT_REGION}" 2>/dev/null || \
awslocal secretsmanager put-secret-value \
  --secret-id "${POSTGRES_SECRET_NAME}" \
  --secret-string "${POSTGRES_URL}" \
  --region "${DEFAULT_REGION}"

awslocal s3 mb "s3://${S3_BUCKET}" --region "${DEFAULT_REGION}" 2>/dev/null || true

echo "LocalStack init complete"
