#!/bin/bash

# Especificar carpetas de entrada y salida
INPUT_FOLDER="data/mp3_25"
OUTPUT_FOLDER="data/example/wavs"

# Crear la carpeta de salida si no existe
mkdir -p "$OUTPUT_FOLDER"

# Inicializar el contador para nombrar secuencialmente
counter=1

# Loop a través de todos los archivos MP3 en la carpeta de entrada
for archivo in "$INPUT_FOLDER"/*.mp3; do
    # Verificar si el archivo existe
    if [[ -f "$archivo" ]]; then
        # Convertir el archivo usando ffmpeg
        ffmpeg -i "$archivo" -acodec pcm_s16le -ar 44100 "$OUTPUT_FOLDER/$counter.wav"
        
        # Incrementar el contador
        counter=$((counter + 1))
    fi
done

echo "Conversión completa. Verifica la carpeta '$OUTPUT_FOLDER' para los archivos convertidos."
