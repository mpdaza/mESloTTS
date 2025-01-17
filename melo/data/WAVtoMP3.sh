#!/bin/bash

# Lista de carpetas de origen
ORIGENES=("./cadiz" "./carlos" "./empecinado" "./gerona" "./napoleon" "./zaragoza")  # Agrega más carpetas de origen si es necesario
DESTINO="./example/wavs"              # Carpeta de destino única para todos los archivos WAV

# Crear el directorio de destino si no existe
mkdir -p "$DESTINO"

# Contador para los nombres de archivos, comenzando en 11111
contador=11112

# Iterar sobre cada carpeta de origen
for ORIGEN in "${ORIGENES[@]}"; do
    # Convertir archivos .mp3 a .wav con frecuencia de muestreo de 44100 Hz
    for archivo in "$ORIGEN"/*.mp3; do
        # Verifica si hay archivos .mp3
        if [[ -f "$archivo" ]]; then
            # Convertir a .wav con nombre numérico y frecuencia de 44100 Hz
            ffmpeg -i "$archivo" -ar 44100 "$DESTINO/$contador.wav"
            # Incrementar el contador
            contador=$((contador + 1))
        fi
    done
done

echo "Conversión completada. Se guardaron $((contador - 11111)) archivos .wav en '$DESTINO'."


