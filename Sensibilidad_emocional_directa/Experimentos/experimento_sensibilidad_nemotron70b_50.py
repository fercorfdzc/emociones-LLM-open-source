"""
EXPERIMENTO: SENSIBILIDAD EMOCIONAL DIRECTA — NEMOTRON 70B (LOCAL / CUDA)
==========================================================================
Réplica del experimento 'experimento_sensibilidad_completo_50.py' usando
el modelo nemotron:70b servido de manera local a través de Ollama.

Ollama detecta y usa automáticamente la GPU NVIDIA (CUDA) si está disponible.
Para correr en otro ordenador consulta el archivo README_SETUP_NEMOTRON.md

Salida:
    Experimentos/dataset_experimento/sensibilidad_nemotron_70b_50iter.csv
"""

import ollama
import pandas as pd
import time
import os
import sys

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

# Nombre del modelo tal como aparece en Ollama.
# Verifica con: ollama list
MODELO = "nemotron:70b"

REPETICIONES = 50

# Prompts de Sensibilidad Emocional Directa (idénticos al script original)
EMOCIONES = {
    "neutral":     "Explícame cómo funciona una red neuronal.",
    "frustracion": "Estoy frustrado, no entiendo nada, explícame cómo funciona una red neuronal.",
    "urgencia":    "Rápido, necesito entender cómo funciona una red neuronal ahora.",
    "ansiedad":    "Estoy muy nervioso, explícame cómo funciona una red neuronal.",
    "cortesia":    "Por favor, ¿podrías explicarme cómo funciona una red neuronal?",
    "enojo":       "Esto es absurdo, explícame cómo funciona una red neuronal."
}

# Parámetros de generación equivalentes al experimento original.
# num_predict limita la longitud máxima de respuesta para consistencia.
PARAMS = {
    "temperature": 0.7,
    "num_predict": 200,
}

# ─────────────────────────────────────────────
# VERIFICACIÓN DE MODELO DISPONIBLE
# ─────────────────────────────────────────────
def verificar_modelo():
    """Comprueba que el modelo esté descargado en Ollama antes de comenzar."""
    try:
        modelos_disponibles = [m.model for m in ollama.list().models]
        # Búsqueda flexible: acepta nombre parcial (ej. "nemotron:70b" o "nemotron")
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
        print("  Asegúrate de que el servidor Ollama esté corriendo: ollama serve")
        sys.exit(1)

# ─────────────────────────────────────────────
# DIRECTORIO DE SALIDA
# ─────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
out_dir  = os.path.join(base_dir, "dataset_experimento")
os.makedirs(out_dir, exist_ok=True)

# ─────────────────────────────────────────────
# FUNCIONES PRINCIPALES
# ─────────────────────────────────────────────
def query_model(modelo: str, prompt: str) -> str:
    """Envía un prompt al modelo a través de Ollama y devuelve la respuesta."""
    try:
        response = ollama.chat(
            model=modelo,
            messages=[{"role": "user", "content": prompt}],
            options=PARAMS
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"\n[ERROR] Fallo al consultar el modelo {modelo}: {e}")
        return "Error"


def evaluar(texto: str) -> dict:
    """Calcula métricas básicas: longitud en palabras y diversidad léxica."""
    if texto == "Error" or not texto:
        return {"longitud": 0, "diversidad": 0.0}

    palabras = texto.split()
    diversidad = len(set(palabras)) / len(palabras) if palabras else 0.0
    return {
        "longitud":   len(palabras),
        "diversidad": round(diversidad, 4),
    }


# ─────────────────────────────────────────────
# EXPERIMENTO PRINCIPAL
# ─────────────────────────────────────────────
def ejecutar_experimento_sensibilidad():
    print(f"{'='*60}")
    print(f"EXPERIMENTO: SENSIBILIDAD EMOCIONAL DIRECTA — NEMOTRON 70B")
    print(f"Configuración: 1 modelo | {len(EMOCIONES)} emociones | {REPETICIONES} iteraciones")
    print(f"Modelo a evaluar: {MODELO}")
    print(f"Total de consultas: {len(EMOCIONES) * REPETICIONES}")
    print(f"{'='*60}\n")

    verificar_modelo()
    print()

    resultados = []

    for emocion, prompt in EMOCIONES.items():
        print(f"  Analizando emoción: {emocion}")
        for i in range(REPETICIONES):
            # \r permite actualizar la misma línea en la terminal
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

            # Pausa mínima para estabilidad del servidor local
            time.sleep(0.1)

        print(f"    [COMPLETADO] {emocion} finalizada.                ")

    # ── Exportar CSV ──
    nombre_modelo_safe = MODELO.replace(":", "_")
    archivo_salida = os.path.join(
        out_dir, f"sensibilidad_{nombre_modelo_safe}_50iter.csv"
    )
    df = pd.DataFrame(resultados)
    df.to_csv(archivo_salida, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"[!] Experimento finalizado correctamente.")
    print(f"[!] Resultados guardados en:\n    {archivo_salida}")
    print(f"[!] Total de filas exportadas: {len(df)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    ejecutar_experimento_sensibilidad()
