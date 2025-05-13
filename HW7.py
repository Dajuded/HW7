# Importar las librerías necesarias
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Verificar si hay una GPU disponible y configurarla
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el conjunto de datos CNN/DailyMail y usar solo una porción de los datos
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset['train'].select(range(1000))  # Usar solo los primeros 1000 ejemplos

# Cargar el modelo y el tokenizador de GPT-2
model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)  # Enviar el modelo a la GPU
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

# Definir el token de relleno usando el token EOS
tokenizer.pad_token = tokenizer.eos_token

# Preprocesar los datos
def preprocess_data(example):
    # Tokenización de las entradas y etiquetas con padding y truncado (con tamaño de secuencia pequeño)
    inputs = tokenizer(example['article'], padding="max_length", truncation=True, max_length=128)  # max_length reducido
    targets = tokenizer(example['highlights'], padding="max_length", truncation=True, max_length=128)  # max_length reducido
    
    # Asegurarse de que los tamaños de las secuencias de entrada y etiquetas sean compatibles
    inputs['labels'] = targets['input_ids']
    return inputs

# Aplicar el preprocesamiento al conjunto de entrenamiento
train_data = train_data.map(preprocess_data, remove_columns=["article", "highlights"])

# Configurar los hiperparámetros del entrenamiento (sin evaluation_strategy)
training_args = TrainingArguments(
    output_dir="./results",              # Directorio de salida para los resultados
    learning_rate=2e-5,                  # Tasa de aprendizaje
    per_device_train_batch_size=1,       # Tamaño de batch para entrenamiento (el más pequeño)
    per_device_eval_batch_size=1,        # Tamaño de batch para evaluación (el más pequeño)
    num_train_epochs=3,                  # Solo 3 épocas para hacer el proceso rápido
    weight_decay=0.01,                   # Decaimiento de peso para evitar sobreajuste
    gradient_accumulation_steps=1,       # No usar gradientes acumulados
    logging_dir="./logs",                # Directorio para los logs
    logging_steps=500,                   # Verificación del estado cada 500 pasos
    save_steps=1000,                     # Guardar los pasos cada 1000 pasos
    report_to="none",                    # Evitar la configuración de reportes a plataformas externas (si es necesario)
)

# Usar Trainer para entrenar el modelo
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

# Entrenar el modelo y obtener los logs
train_results = trainer.train()

# Acceder a los resultados de entrenamiento desde los logs
logs = trainer.state.log_history[-1]  # Último registro del historial de logs

# Guardar los resultados del entrenamiento en un archivo de texto
with open('training_results.txt', 'w') as f:
    f.write(f"Training Loss: {logs.get('loss')}\n")
    f.write(f"Training Runtime: {logs.get('step_time') * logs.get('num_train_epochs')} seconds\n")
    f.write(f"Training Samples per second: {logs.get('samples_per_second')}\n")
    f.write(f"Training Steps per second: {logs.get('steps_per_second')}\n")
