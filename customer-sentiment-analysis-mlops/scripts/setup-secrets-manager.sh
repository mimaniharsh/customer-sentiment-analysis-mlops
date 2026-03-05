#!/bin/bash
# ================================================================
# Setup AWS Secrets Manager for Sentient Platform (Production)
# Run this ONCE to store your credentials securely.
# Never run this with credentials visible in shell history.
# ================================================================

set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
SECRET_NAME="sentient/production/secrets"

echo "Creating secret in AWS Secrets Manager: $SECRET_NAME"
echo "Enter your secret values when prompted..."

read -rsp "AWS_ACCESS_KEY_ID: " ACCESS_KEY; echo
read -rsp "AWS_SECRET_ACCESS_KEY: " SECRET_KEY; echo
read -rsp "APP_SECRET_KEY (press enter to generate): " APP_KEY; echo
read -rsp "JWT_SECRET_KEY (press enter to generate): " JWT_KEY; echo
read -rsp "DATABASE_URL: " DB_URL; echo

APP_KEY="${APP_KEY:-$(openssl rand -hex 32)}"
JWT_KEY="${JWT_KEY:-$(openssl rand -hex 32)}"

SECRET_JSON=$(jq -n \
  --arg ak "$ACCESS_KEY" \
  --arg sk "$SECRET_KEY" \
  --arg appk "$APP_KEY" \
  --arg jwtk "$JWT_KEY" \
  --arg db "$DB_URL" \
  '{
    AWS_ACCESS_KEY_ID: $ak,
    AWS_SECRET_ACCESS_KEY: $sk,
    APP_SECRET_KEY: $appk,
    JWT_SECRET_KEY: $jwtk,
    DATABASE_URL: $db
  }')

aws secretsmanager create-secret \
  --name "$SECRET_NAME" \
  --description "Sentient Platform production secrets" \
  --secret-string "$SECRET_JSON" \
  --region "$REGION" 2>/dev/null || \
aws secretsmanager update-secret \
  --secret-id "$SECRET_NAME" \
  --secret-string "$SECRET_JSON" \
  --region "$REGION"

echo ""
echo "✅ Secrets stored in AWS Secrets Manager: $SECRET_NAME"
echo "   Set APP_ENV=production in your ECS task definition."
echo "   Ensure the task execution role has secretsmanager:GetSecretValue"

# Unset from shell
unset ACCESS_KEY SECRET_KEY APP_KEY JWT_KEY DB_URL SECRET_JSON
