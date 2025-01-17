import os
from pydub import AudioSegment

# Función para resamplear y guardar los archivos
def resample_audio(input_folders, output_folder):
    # Crea la carpeta de salida si no existe (incluyendo los directorios padres)
    os.makedirs(output_folder, exist_ok=True)
    
    # Variable para contar el número de archivos procesados
    count = 1
    
    # Procesa cada carpeta de entrada
    for input_folder in input_folders:
        # Verifica que la carpeta de entrada exista
        if not os.path.exists(input_folder):
            print(f"Error: La carpeta de entrada '{input_folder}' no existe.")
            continue
        
        # Lista los archivos en la carpeta de entrada
        files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
        
        # Procesa cada archivo en la carpeta de entrada
        for file_name in files:
            file_path = os.path.join(input_folder, file_name)
            
            # Intenta cargar el archivo de audio
            try:
                # Carga el audio usando pydub
                audio = AudioSegment.from_file(file_path)
                
                # Cambia la frecuencia de muestreo a 44100 Hz
                audio = audio.set_frame_rate(44100)
                
                # Exporta el archivo en formato WAV y con el nombre "count.wav"
                output_path = os.path.join(output_folder, f"{count}.wav")
                audio.export(output_path, format="wav")
                
                print(f"Procesado: {file_name} -> {count}.wav")
                count += 1
            except Exception as e:
                print(f"Error procesando {file_name}: {e}")

if __name__ == "__main__":
    # Lista de carpetas de entrada
    input_folders = ["data/karen/karen/big_time_1109_librivox", "data/karen/karen/elusive_pimpernel_krs_librivox_64kb_mp3", "data/karen/karen/mansfield_park_1104_librivox"]
    output_folder = "data/example/wavs"

    # Validar que las rutas no están vacías
    if not input_folders or not output_folder:
        print("Error: Debes introducir rutas válidas tanto para las carpetas de entrada como para la de salida.")
    else:
        # Ejecuta la función para resamplear los audios
        resample_audio(input_folders, output_folder)
