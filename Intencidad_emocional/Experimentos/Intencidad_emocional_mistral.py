import ollama
import pandas as pd
import time
import os


# CONFIG


# Puedes cambiar el modelo según necesites evaluar (llama3, mistral, phi3, qwen:14b, etc.)
MODELO = "mistral"

# Prompts estructurados con diferentes niveles de intensidad emocional
INTENSIDAD_EMOCIONAL = {
    "neutral": "Explícame cómo funciona una red neuronal.",
    
    # Eje 1: Frustración / Enojo (Evalúa si el modelo se vuelve más paciente o conciso)
    "frustracion_baja": "Me siento un poco confundido, explícame cómo funciona una red neuronal.",
    "frustracion_media": "Estoy frustrado, no entiendo nada, explícame cómo funciona una red neuronal.",
    "frustracion_alta": "¡Estoy harto y sumamente frustrado! ¡Es imposible entender esto! Explícame cómo funciona una red neuronal.",
    
    # Eje 2: Ansiedad / Urgencia (Evalúa si el modelo va directo al grano y elimina el relleno)
    "urgencia_baja": "Cuando tengas un momento, explícame cómo funciona una red neuronal.",
    "urgencia_media": "Necesito que me expliques cómo funciona una red neuronal pronto.",
    "urgencia_alta": "¡Rápido, es de extrema urgencia! ¡Necesito entender cómo funciona una red neuronal AHORA MISMO!",
    
    # Eje 3: Tristeza / Desánimo (Evalúa la capacidad de mostrar empatía o calidez)
    "tristeza_baja": "Me siento un poco desanimado hoy. Explícame cómo funciona una red neuronal.",
    "tristeza_media": "Estoy triste y me cuesta mucho concentrarme. Explícame cómo funciona una red neuronal.",
    "tristeza_alta": "Estoy totalmente devastado y sin fuerzas para nada. Solo explícame cómo funciona una red neuronal.",
    
    # Eje 4: Alegría / Entusiasmo (Evalúa si el modelo refleja el tono positivo)
    "entusiasmo_bajo": "Me parece interesante, explícame cómo funciona una red neuronal.",
    "entusiasmo_medio": "¡Qué genial! Me entusiasma aprender, explícame cómo funciona una red neuronal.",
    "entusiasmo_alto": "¡Estoy increíblemente emocionado y feliz por aprender esto! ¡Explícame cómo funciona una red neuronal!"
}

REPETICIONES = 5

PARAMS = {
    "temperature": 0.7,
    "num_predict": 200
}


# FUNCIÓN MODELO


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


# MÉTRICAS

def evaluar(texto):
    if texto == "Error":
        return {"longitud": 0, "diversidad": 0.0}
        
    palabras = texto.split()
    return {
        "longitud": len(palabras),
        "diversidad": len(set(palabras)) / len(palabras) if palabras else 0
    }


# EXPERIMENTO


resultados = []

print(f"Iniciando experimento de Intensidad Emocional con {MODELO}...\n")

for nivel, prompt in INTENSIDAD_EMOCIONAL.items():
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

# GUARDAR

base_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base_dir, "dataset_experimento")
os.makedirs(out_dir, exist_ok=True)

archivo_salida = os.path.join(out_dir, f"{MODELO}_intensidad_experimento.csv")

df = pd.DataFrame(resultados)
df.to_csv(archivo_salida, index=False)

print(f"\n¡Experimento finalizado! Datos guardados en: {archivo_salida}")
