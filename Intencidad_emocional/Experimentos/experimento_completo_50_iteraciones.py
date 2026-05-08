import ollama
import pandas as pd
import time
import os

# CONFIGURACIÓN
# Lista de los 4 modelos a iterar. Puedes ajustarlos según los que tengas instalados.
MODELOS = ["llama3", "mistral", "phi3", "llama3.2"]
REPETICIONES = 50

# Prompts estructurados con diferentes niveles de intensidad emocional
INTENSIDAD_EMOCIONAL = {
    "neutral": "Explícame cómo funciona una red neuronal.",
    
    # Eje 1: Frustración / Enojo
    "frustracion_baja": "Me siento un poco confundido, explícame cómo funciona una red neuronal.",
    "frustracion_media": "Estoy frustrado, no entiendo nada, explícame cómo funciona una red neuronal.",
    "frustracion_alta": "¡Estoy harto y sumamente frustrado! ¡Es imposible entender esto! Explícame cómo funciona una red neuronal.",
    
    # Eje 2: Ansiedad / Urgencia
    "urgencia_baja": "Cuando tengas un momento, explícame cómo funciona una red neuronal.",
    "urgencia_media": "Necesito que me expliques cómo funciona una red neuronal pronto.",
    "urgencia_alta": "¡Rápido, es de extrema urgencia! ¡Necesito entender cómo funciona una red neuronal AHORA MISMO!",
    
    # Eje 3: Tristeza / Desánimo
    "tristeza_baja": "Me siento un poco desanimado hoy. Explícame cómo funciona una red neuronal.",
    "tristeza_media": "Estoy triste y me cuesta mucho concentrarme. Explícame cómo funciona una red neuronal.",
    "tristeza_alta": "Estoy totalmente devastado y sin fuerzas para nada. Solo explícame cómo funciona una red neuronal.",
    
    # Eje 4: Alegría / Entusiasmo
    "entusiasmo_bajo": "Me parece interesante, explícame cómo funciona una red neuronal.",
    "entusiasmo_medio": "¡Qué genial! Me entusiasma aprender, explícame cómo funciona una red neuronal.",
    "entusiasmo_alto": "¡Estoy increíblemente emocionado y feliz por aprender esto! ¡Explícame cómo funciona una red neuronal!"
}

# Parámetros de generación
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

# FUNCIÓN PARA EVALUAR MÉTRICAS BÁSICAS
def evaluar(texto):
    if texto == "Error" or not texto:
        return {"longitud": 0, "diversidad": 0.0}
        
    palabras = texto.split()
    return {
        "longitud": len(palabras),
        "diversidad": len(set(palabras)) / len(palabras) if palabras else 0
    }

# DIRECTORIO DE SALIDA
# Se guardará en una carpeta llamada 'dataset_experimento' en el mismo lugar que este script
base_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base_dir, "dataset_experimento")
os.makedirs(out_dir, exist_ok=True)

def ejecutar_experimento():
    print(f"{'='*60}")
    print(f"INICIANDO EXPERIMENTO AUTOMATIZADO")
    print(f"Configuración: {len(MODELOS)} modelos | {len(INTENSIDAD_EMOCIONAL)} prompts | {REPETICIONES} iteraciones")
    print(f"Modelos: {', '.join(MODELOS)}")
    print(f"{'='*60}\n")

    for modelo in MODELOS:
        print(f">>> PROCESANDO MODELO: {modelo}")
        resultados_modelo = []
        
        # Iniciamos el bucle por cada nivel de intensidad
        for nivel, prompt in INTENSIDAD_EMOCIONAL.items():
            print(f"  Nivel: {nivel}")
            for i in range(REPETICIONES):
                # Usamos \r para que la línea se actualice en la misma posición
                print(f"    Iteración {i+1}/{REPETICIONES}...", end="\r")
                
                respuesta = query_model(modelo, prompt)
                metricas = evaluar(respuesta)
                
                resultados_modelo.append({
                    "modelo": modelo,
                    "intensidad": nivel,
                    "iteracion": i + 1,
                    "prompt": prompt,
                    "respuesta": respuesta,
                    **metricas
                })
                
                # Pequeña pausa opcional para estabilidad
                time.sleep(0.1)
            print(f"    [OK] {nivel} completado.                ")

        # Guardar el CSV individual del modelo al terminar sus 50 iteraciones
        archivo_salida = os.path.join(out_dir, f"resultado_{modelo.replace(':', '_')}_50iter.csv")
        df = pd.DataFrame(resultados_modelo)
        df.to_csv(archivo_salida, index=False)
        
        print(f"\n[!] {modelo} finalizado. Datos guardados en: {archivo_salida}")
        print(f"{'-'*60}\n")

    print(f"{'='*60}")
    print("¡EXPERIMENTO COMPLETO FINALIZADO EXITOSAMENTE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    ejecutar_experimento()
