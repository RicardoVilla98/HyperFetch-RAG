#!/bin/bash
# ==============================================================================
# Helper script to link and push HyperFetch-RAG to your GitHub account
# ==============================================================================

set -e

if [ -z "$1" ]; then
    echo "⚠️  Uso: ./push_to_github.sh <TU_USUARIO_GITHUB> [protocolo: https o ssh]"
    echo "Ejemplo HTTPS: ./push_to_github.sh miusuario"
    echo "Ejemplo SSH:   ./push_to_github.sh miusuario ssh"
    exit 1
fi

GITHUB_USER="$1"
REPO_NAME="HyperFetch-RAG"
PROTOCOL="${2:-https}"

if [ "$PROTOCOL" = "ssh" ]; then
    REMOTE_URL="git@github.com:${GITHUB_USER}/${REPO_NAME}.git"
else
    REMOTE_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
fi

echo "🚀 Configurando remote origin: $REMOTE_URL"

# Remove existing origin if any
git remote remove origin 2>/dev/null || true

# Add new origin
git remote add origin "$REMOTE_URL"

echo "📦 Subiendo rama main a GitHub..."
git push -u origin main

echo "✅ Repositorio subido exitosamente a: https://github.com/${GITHUB_USER}/${REPO_NAME}"
