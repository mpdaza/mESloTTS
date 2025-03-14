#!/bin/bash
set -e

TEXT="Esto es una prueba del funcionamiento del tts"
MODEL="/models"
OUTPUT="/output"

# Check if required environment variables are set
if [ -z "$TEXT" ]; then
    echo "ERROR: TEXT environment variable is not set"
    exit 1
fi

if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL environment variable is not set"
    exit 1
fi

if [ -z "$OUTPUT" ]; then
    echo "ERROR: OUTPUT environment variable is not set"
    exit 1
fi

# Run the inference command with the environment variables
python /app/melo/infer.py --text "$TEXT" -m "$MODEL" -o "$OUTPUT"