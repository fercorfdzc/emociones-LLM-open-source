import ollama
import pandas as pd
import time
import os

# CONFIGURACIÓN
MODELO = "llama3"
REPETICIONES = 3 # Número de veces que se evalúa cada prompt
PARAMS = {
    "temperature": 0.7,
    "num_predict": 200
}

# ARCHIVO DE ENTRADA CON LA "NUEVA DATASET" DE PROMPTS
# Se espera un CSV con columnas: 'intensidad' y 'prompt'
base_dir = os.path.dirname(os.path.abspath(__file__))
archivo_entrada = os.path.join(base_dir, "nuevos_prompts_emocionales.csv")
archivo_salida = os.path.join(base_dir, "..", "dataset_experimento", f"{MODELO}_nueva_intensidad.csv")

def query_model(prompt):
    try:
        response = ollama.chat(
            model=MODELO,
            messages=[{"role": "user", "content": prompt}],
            options=PARAMS
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"Error al consultar el modelo: {e}")
        return "Error"

def evaluar(texto):
    if texto == "Error":
        return {"longitud": 0, "diversidad": 0.0}
    palabras = texto.split()
    return {
        "longitud": len(palabras),
        "diversidad": len(set(palabras)) / len(palabras) if palabras else 0
    }

def main():
    if not os.path.exists(archivo_entrada):
        print(f"Error: No se encontró el archivo de entrada '{archivo_entrada}'.")
        print("Asegúrate de crear un archivo CSV con las columnas 'intensidad' y 'prompt'.")
        # Creamos un CSV de ejemplo si no existe
        df_ejemplo = pd.DataFrame([
            {"intensidad": "nueva_frustracion_baja", "prompt": "No entiendo muy bien esto."},
            {"intensidad": "nueva_frustracion_alta", "prompt": "¡Esto es totalmente incomprensible y me rindo!"}
        ])
        df_ejemplo.to_csv(archivo_entrada, index=False)
        print(f"Se ha creado un archivo de ejemplo en: {archivo_entrada}")
        return

    print(f"Leyendo prompts desde: {archivo_entrada}")
    df_prompts = pd.read_csv(archivo_entrada)
    
    if 'intensidad' not in df_prompts.columns or 'prompt' not in df_prompts.columns:
        print("El archivo CSV debe contener las columnas 'intensidad' y 'prompt'.")
        return

    resultados = []
    print(f"\nIniciando experimento de Intensidad Emocional con {MODELO}...\n")

    for index, row in df_prompts.iterrows():
        nivel = row['intensidad']
        prompt = row['prompt']
        
        for i in range(REPETICIONES):
            print(f"{nivel} | iter {i+1}")
            respuesta = query_model(prompt)
            print(f"Respuesta:\n{respuesta}\n{'-'*40}")
            metricas = evaluar(respuesta)

            resultados.append({
                "intensidad": nivel,
                "iteracion": i,
                "prompt": prompt,
                "respuesta": respuesta,
                **metricas
            })
            time.sleep(1)

    # Guardar resultados
    os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(archivo_salida, index=False)
    print(f"\n¡Experimento finalizado! Datos guardados en: {archivo_salida}")

if __name__ == "__main__":
    main()
