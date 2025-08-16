#!/bin/bash

# Securely start the API server with Groq key from keychain
cd /Users/tem/rho/api

# Retrieve Groq API key from keychain
GROQ_API_KEY=$(security find-generic-password -s "groq API key" -w 2>/dev/null)

if [ -z "$GROQ_API_KEY" ]; then
    echo "Warning: Groq API key not found in keychain. Some features may not work."
    echo "To add it, run: security add-generic-password -s 'groq API key' -w"
    export GROQ_API_KEY=""
else
    export GROQ_API_KEY="$GROQ_API_KEY"
    echo "✅ Groq API key loaded from keychain"
fi

# Start the server
poetry run uvicorn main:app --reload --port 8192