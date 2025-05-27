# Importar las librerías necesarias
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Verificar si hay una GPU disponible y configurarla
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo y el tokenizador ya entrenado
model = GPT2LMHeadModel.from_pretrained("./results").to(device)  # Cargar el modelo guardado
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

# Definir el token de relleno usando el token EOS
tokenizer.pad_token = tokenizer.eos_token

# Función para resumir un texto
def summarize_text(text, max_input_length=512, max_output_length=150):
    # Tokenización del texto de entrada
    inputs = tokenizer(text, return_tensors="pt", max_length=max_input_length, truncation=True, padding=True).to(device)

    # Generar el resumen utilizando el modelo
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=max_output_length, 
        num_beams=4,                  # Usar beam search para una mejor calidad
        no_repeat_ngram_size=2,       # Evitar la repetición de n-gramas
        early_stopping=True,          # Detener temprano cuando se haya generado el resumen
        length_penalty=2.0,           # Penalizar longitudes de resumen excesivas
        top_k=50,                     # Número de palabras principales a considerar para la generación
        top_p=0.95                    # Muestra de palabras con probabilidad acumulada
    )

    # Decodificar el resumen generado
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Función para pedir el texto al usuario
def get_input_text():
    print("Introduce el texto que deseas resumir:")
    input_text = ""
    while True:
        line = input()
        if line:
            input_text += line + " "
        else:
            break
    return input_text

# Obtener el texto desde el usuario
input_text = get_input_text()

# Obtener el resumen
summary = summarize_text(input_text)
print("\nResumen generado:")
print(summary)
