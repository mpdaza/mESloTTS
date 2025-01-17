import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# Ruta de las carpetas de origen y la carpeta de destino
source_folders = ["./female_karen/big_time_1109_librivox","./female_karen/elusive_pimpernel_krs_librivox_64kb_mp3", "./female_karen/mansfield_park_1104_librivox"]
destination_folder = "./example/wavs"

# source_folders = ["./cadiz", "./carlos", "./empecinado", "./gerona", "./napoleon", "./zaragoza"]
# destination_folder = "./new_wavs"

# Duración máxima del fragmento en milisegundos (10-15 segundos)
min_duration = 10000  # 10 segundos
max_duration = 15000  # 15 segundos

# Frecuencia de muestreo en Hz
sample_rate = 41000

# Contador inicial para el nombre del archivo
file_counter = 1

# Función para procesar y dividir el audio en fragmentos
def process_audio(file_path, destination_folder, file_counter):
    # Carga el archivo de audio
    audio = AudioSegment.from_file(file_path)
    
    # Posición de inicio para el primer fragmento
    start = 0
    
    # Divide el audio en fragmentos de 15 segundos (o menos, sin cortar palabras)
    while start < len(audio):
        # Define el punto final tentativo del fragmento
        end = start + max_duration

        # Asegura que el fragmento no exceda la duración total del audio
        if end > len(audio):
            end = len(audio)

        # Busca el final de la palabra o frase en el intervalo (si es posible)
        nonsilent_ranges = detect_nonsilent(audio[start:end], min_silence_len=200, silence_thresh=-40)

        # Ajusta el punto de corte al final de la última palabra detectada
        if nonsilent_ranges:
            last_nonsilent_end = nonsilent_ranges[-1][1]
            if last_nonsilent_end >= min_duration:
                end = start + last_nonsilent_end
        
        # Extrae el fragmento de audio
        audio_chunk = audio[start:end]

        # Configura la frecuencia de muestreo a 41000 Hz
        audio_chunk = audio_chunk.set_frame_rate(sample_rate)

        # Asegura la carpeta de destino
        os.makedirs(destination_folder, exist_ok=True)

        # Guarda el fragmento con un nombre de archivo secuencial
        file_name = f"{file_counter}.wav"
        destination_path = os.path.join(destination_folder, file_name)
        audio_chunk.export(destination_path, format="wav")
        print(f"Archivo procesado y guardado: {destination_path}")

        # Incrementa el contador y ajusta el inicio para el siguiente fragmento
        file_counter += 1
        start = end

    return file_counter  # Devuelve el contador actualizado para el siguiente archivo

# Procesa cada carpeta de origen
for folder in source_folders:
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.lower().endswith(('.mp3', '.ogg', '.flac', '.wav')):  # formatos de entrada
                file_counter = process_audio(file_path, destination_folder, file_counter)
