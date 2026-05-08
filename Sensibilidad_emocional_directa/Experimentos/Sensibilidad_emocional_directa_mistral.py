import ollama
import pandas as pd
import time
import os

# CONFIG

MODELO = "mistral"

EMOCIONES = {
    "neutral": "Explícame cómo funciona una red neuronal.",
    "frustracion": "Estoy frustrado, no entiendo nada, explícame cómo funciona una red neuronal.",
    "urgencia": "Rápido, necesito entender cómo funciona una red neuronal ahora.",
    "ansiedad": "Estoy muy nervioso, explícame cómo funciona una red neuronal.",
    "cortesia": "Por favor, ¿podrías explicarme cómo funciona una red neuronal?",
    "enojo": "Esto es absurdo, explícame cómo funciona una red neuronal."
}

REPETICIONES = 5

PARAMS = {
    "temperature": 0.7,
    "num_predict": 200
}

# FUNCIÓN MODELO

def query_model(prompt):
    response = ollama.chat(
        model=MODELO,
        messages=[{"role": "user", "content": prompt}],
        options=PARAMS
    )
    return response["message"]["content"]

# ==============================
# MÉTRICAS
# ==============================

def evaluar(texto):
    palabras = texto.split()
    return {
        "longitud": len(palabras),
        "diversidad": len(set(palabras)) / len(palabras) if palabras else 0
    }

# ==============================
# EXPERIMENTO
# ==============================

resultados = []

for emocion, prompt in EMOCIONES.items():
    for i in range(REPETICIONES):

        print(f"{emocion} | iter {i+1}")

        respuesta = query_model(prompt)
        print(f"Respuesta:\n{respuesta}\n{'-'*40}")
        metricas = evaluar(respuesta)

        resultados.append({
            "emocion": emocion,
            "iteracion": i,
            "prompt": prompt,
            "respuesta": respuesta,
            **metricas
        })

        time.sleep(1)

# ==============================
# GUARDAR
# ==============================

# Aseguramos que la carpeta exista si se desea guardar ahí
os.makedirs("dataset_experimento", exist_ok=True)

df = pd.DataFrame(resultados)
df.to_csv("dataset_experimento/mistral_experimento.csv", index=False)

print("Listo")
