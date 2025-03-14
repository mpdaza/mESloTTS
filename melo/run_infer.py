import os
import csv

# Parameters
modelo = "/home/mp/Documentos/mESloTTS/melo/data/example/mateo/models/G_1500.pth"
salida = "/home/mp/Documentos/mESloTTS/melo/data/example/mateo/outputs"
archivo_textos = "/home/mp/Documentos/mESloTTS/melo/data/example/mateo/sentences.txt"


# Leer los textos del archivo
with open(archivo_textos, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=';')
    textos = list(reader)

# Recorrer los textos y ejecutar el comando para cada uno
for i, row in enumerate(textos):
    texto = row['text'].strip()
    if texto:  # Verifica que no esté vacío
        nombre_salida = f"output{i}_{row['text']}"
        comando = f"python melo/infer.py --text \"{row['sentence']}\" -m {modelo} -o {os.path.join(salida, nombre_salida)}"
        os.system(comando)  # Ejecuta el comando en la terminal

print("Procesamiento completado.")