import os
from pydub import AudioSegment

# Función para resamplear y guardar los archivos
def resample_audio(input_folder, output_folder):
    # Verifica si la carpeta de salida existe, si no, la crea
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Lista los archivos en la carpeta de entrada
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    # Variable para contar el número de archivos procesados
    count = 1
    
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
    # Define la carpeta de entrada y salida
    input_folder = input("./data/NUFA_Albayzin01/")
    output_folder = input("./prueba")
    
    # Ejecuta la función para resamplear los audios
    resample_audio(input_folder, output_folder)
