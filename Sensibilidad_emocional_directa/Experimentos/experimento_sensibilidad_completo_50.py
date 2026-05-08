import ollama
import pandas as pd
import time
import os

# CONFIGURACIÓN
# Lista de los 4 modelos a iterar secuencialmente
MODELOS = ["llama3", "mistral", "phi3","qwen:14b"]
REPETICIONES = 50

# Prompts de Sensibilidad Emocional Directa (Basado en el script original)
EMOCIONES = {
    "neutral": "Explícame cómo funciona una red neuronal.",
    "frustracion": "Estoy frustrado, no entiendo nada, explícame cómo funciona una red neuronal.",
    "urgencia": "Rápido, necesito entender cómo funciona una red neuronal ahora.",
    "ansiedad": "Estoy muy nervioso, explícame cómo funciona una red neuronal.",
    "cortesia": "Por favor, ¿podrías explicarme cómo funciona una red neuronal?",
    "enojo": "Esto es absurdo, explícame cómo funciona una red neuronal."
}

# Parámetros para la generación del modelo
PARAMS = {
    "temperature": 0.7,
    "num_predict": 200
}

# FUNCIÓN PARA CONSULTAR EL MODELO
def query_model(modelo, prompt):
    try:
        response = ollama.chat(
            model=modelo,
            messages=[{"role": "user", "content": prompt}],
            options=PARAMS
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"\nError al consultar el modelo {modelo}: {e}")
        return "Error"

# FUNCIÓN PARA EVALUAR MÉTRICAS (Basado en el script original)
def evaluar(texto):
    if texto == "Error" or not texto:
        return {"longitud": 0, "diversidad": 0.0}
        
    palabras = texto.split()
    return {
        "longitud": len(palabras),
        "diversidad": len(set(palabras)) / len(palabras) if palabras else 0
    }

# DIRECTORIO DE SALIDA
# Se guardará en 'dataset_experimento' dentro de la carpeta del proyecto
base_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base_dir, "dataset_experimento")
os.makedirs(out_dir, exist_ok=True)

def ejecutar_experimento_sensibilidad():
    print(f"{'='*60}")
    print(f"EXPERIMENTO: SENSIBILIDAD EMOCIONAL DIRECTA (MULTIMODELO)")
    print(f"Configuración: {len(MODELOS)} modelos | {len(EMOCIONES)} emociones | {REPETICIONES} iteraciones")
    print(f"Modelos a evaluar: {', '.join(MODELOS)}")
    print(f"{'='*60}\n")

    for modelo in MODELOS:
        print(f"===> INICIANDO EVALUACIÓN DE: {modelo}")
        resultados_modelo = []
        
        for emocion, prompt in EMOCIONES.items():
            print(f"  Analizando emoción: {emocion}")
            for i in range(REPETICIONES):
                # \r permite actualizar la misma línea en la terminal
                print(f"    Iteración {i+1}/{REPETICIONES}...", end="\r")
                
                respuesta = query_model(modelo, prompt)
                metricas = evaluar(respuesta)
                
                resultados_modelo.append({
                    "modelo": modelo,
                    "emocion": emocion,
                    "iteracion": i + 1,
                    "prompt": prompt,
                    "respuesta": respuesta,
                    **metricas
                })
                
                # Pausa mínima para estabilidad
                time.sleep(0.1)
            print(f"    [COMPLETADO] {emocion} finalizada.                ")

        # Guardar CSV por cada modelo para asegurar los datos
        archivo_salida = os.path.join(out_dir, f"sensibilidad_{modelo.replace(':', '_')}_50iter.csv")
        df = pd.DataFrame(resultados_modelo)
        df.to_csv(archivo_salida, index=False)
        
        print(f"\n[!] Datos de {modelo} exportados a: {archivo_salida}")
        print(f"{'-'*60}\n")

    print(f"{'='*60}")
    print("¡PROCESO FINALIZADO PARA TODOS LOS MODELOS!")
    print(f"{'='*60}")

if __name__ == "__main__":
    ejecutar_experimento_sensibilidad()
