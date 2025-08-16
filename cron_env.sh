#!/bin/bash
# Retrieve API key from macOS keychain
GROQ_API_KEY=$(security find-generic-password -s "groq API key" -w 2>/dev/null)
export GROQ_API_KEY="$GROQ_API_KEY"
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
cd "/Users/tem/rho"
