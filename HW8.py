import re

def simple_simulated_summary(text, max_sentences=3):
    # Separar el texto en oraciones (muy básico)
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    
    # Simular "resumen" tomando las primeras max_sentences oraciones
    # (como CNN/DailyMail donde resumen típicamente son las primeras ideas principales)
    summary = ' '.join(sentences[:max_sentences])
    
    return summary

if __name__ == "__main__":
    print("Simulador simple de resumen estilo CNN/DailyMail (sin librerías externas).")
    while True:
        text = input("\nIngrese texto en inglés (o 'exit' para salir):\n")
        if text.lower() == "exit":
            print("Programa terminado.")
            break
        resumen_simulado = simple_simulated_summary(text)
        print("\nResumen simulado:\n", resumen_simulado)
        