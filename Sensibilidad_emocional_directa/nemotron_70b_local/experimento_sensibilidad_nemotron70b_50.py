"""
EXPERIMENTO: SENSIBILIDAD EMOCIONAL DIRECTA — NEMOTRON 70B (LOCAL / CUDA)
==========================================================================
Réplica del experimento de sensibilidad emocional directa usando
el modelo nemotron:70b servido localmente a través de Ollama.

Ollama detecta y usa automáticamente la GPU NVIDIA (CUDA) si está disponible.
Consulta README_SETUP.md para instrucciones de configuración.

Salida:
    dataset_experimento/sensibilidad_nemotron_70b_50iter.csv
"""

import ollama
import pandas as pd
import time
import os
import sys


MODELO = "nemotron:70b"

REPETICIONES = 50

# Prompts de Sensibilidad Emocional Directa
EMOCIONES = {
    "neutral":     "Explícame cómo funciona una red neuronal.",
    "frustracion": "Estoy frustrado, no entiendo nada, explícame cómo funciona una red neuronal.",
    "urgencia":    "Rápido, necesito entender cómo funciona una red neuronal ahora.",
    "ansiedad":    "Estoy muy nervioso, explícame cómo funciona una red neuronal.",
    "cortesia":    "Por favor, ¿podrías explicarme cómo funciona una red neuronal?",
    "enojo":       "Esto es absurdo, explícame cómo funciona una red neuronal."
}

# Parámetros de generación
PARAMS = {
    "temperature": 0.7,
    "num_predict": 200,
}


base_dir = os.path.dirname(os.path.abspath(__file__))
out_dir  = os.path.join(base_dir, "dataset_experimento")
os.makedirs(out_dir, exist_ok=True)


def verificar_modelo():
    """Comprueba que Ollama esté corriendo y el modelo esté descargado."""
    try:
        modelos_disponibles = [m.model for m in ollama.list().models]
        disponible = any(MODELO in m for m in modelos_disponibles)
        if not disponible:
            print(f"\n[ADVERTENCIA] El modelo '{MODELO}' no se encontró en Ollama.")
            print(f"  Modelos disponibles: {modelos_disponibles}")
            print(f"\n  Descárgalo con:\n    ollama pull {MODELO}\n")
            respuesta = input("  ¿Deseas continuar de todos modos? (s/N): ").strip().lower()
            if respuesta != "s":
                print("  Experimento cancelado.")
                sys.exit(1)
        else:
            print(f"[OK] Modelo '{MODELO}' disponible en Ollama.")
    except Exception as e:
        print(f"\n[ERROR] No se pudo conectar con Ollama: {e}")
        print("  Asegúrate de que el servidor esté corriendo: ollama serve")
        sys.exit(1)


def query_model(modelo: str, prompt: str) -> str:
    """Consulta el modelo vía Ollama y devuelve la respuesta como texto."""
    try:
        response = ollama.chat(
            model=modelo,
            messages=[{"role": "user", "content": prompt}],
            options=PARAMS
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"\n[ERROR] Fallo al consultar {modelo}: {e}")
        return "Error"


def evaluar(texto: str) -> dict:
    """Calcula longitud (palabras) y diversidad léxica de la respuesta."""
    if texto == "Error" or not texto:
        return {"longitud": 0, "diversidad": 0.0}
    palabras = texto.split()
    diversidad = len(set(palabras)) / len(palabras) if palabras else 0.0
    return {
        "longitud":   len(palabras),
        "diversidad": round(diversidad, 4),
    }


def ejecutar_experimento_sensibilidad():
    print(f"{'='*60}")
    print(f"EXPERIMENTO: SENSIBILIDAD EMOCIONAL DIRECTA — NEMOTRON 70B")
    print(f"Configuración: 1 modelo | {len(EMOCIONES)} emociones | {REPETICIONES} iteraciones")
    print(f"Modelo: {MODELO}")
    print(f"Total de consultas: {len(EMOCIONES) * REPETICIONES}")
    print(f"{'='*60}\n")

    verificar_modelo()
    print()

    resultados = []

    for emocion, prompt in EMOCIONES.items():
        print(f"  Analizando emoción: {emocion}")
        for i in range(REPETICIONES):
            print(f"    Iteración {i+1}/{REPETICIONES}...", end="\r")

            respuesta = query_model(MODELO, prompt)
            metricas  = evaluar(respuesta)

            resultados.append({
                "modelo":    MODELO,
                "emocion":   emocion,
                "iteracion": i + 1,
                "prompt":    prompt,
                "respuesta": respuesta,
                **metricas,
            })

            time.sleep(0.1)

        print(f"    [COMPLETADO] {emocion} finalizada.                ")

    # ── Exportar CSV ──
    nombre_modelo_safe = MODELO.replace(":", "_")
    archivo_salida = os.path.join(out_dir, f"sensibilidad_{nombre_modelo_safe}_50iter.csv")
    df = pd.DataFrame(resultados)
    df.to_csv(archivo_salida, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"[!] Experimento finalizado correctamente.")
    print(f"[!] Resultados en:\n    {archivo_salida}")
    print(f"[!] Total de filas: {len(df)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    ejecutar_experimento_sensibilidad()
