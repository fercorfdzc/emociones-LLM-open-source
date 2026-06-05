# Evaluación de Emociones en LLMs usando MMLU

Este repositorio contiene la infraestructura metodológica para evaluar cómo los estímulos emocionales afectan el rendimiento de Modelos de Lenguaje Grande (LLMs) como Nemotron, utilizando el benchmark estandarizado MMLU.

## Justificación Metodológica (Para el Paper)

El diseño de este experimento está estrictamente fundamentado en el "Gold Standard" de la literatura científica sobre la sensibilidad emocional de los LLMs. Específicamente, seguimos la metodología conocida como **EmotionPrompt**, descrita y validada en el siguiente artículo científico clave:

> **Li, C., Wang, J., Zhang, Y., Zhu, K., Hou, W., Lian, J., Luo, F., Yang, Q., & Xie, X. (2023).**
> *"Large Language Models Understand and Can be Enhanced by Emotional Stimuli."*
> arXiv preprint arXiv:2307.11760.
> [Enlace al Paper](https://arxiv.org/abs/2307.11760)

### ¿Por qué EmotionPrompt en lugar de otras técnicas (como GoEmotions)?

1. **Preservación de la Validez del Benchmark:** MMLU es una prueba estandarizada (Hendrycks et al., 2021). Si alteráramos la semántica de la pregunta inyectando comentarios externos (por ejemplo, textos aleatorios de Reddit como los del dataset *GoEmotions*) o usando LLMs para reescribir las preguntas, introduciríamos variables de confusión graves. El modelo podría fallar la pregunta no por estar bajo la influencia de una emoción, sino porque la información ajena a la tarea lo distrajo.
2. **Estímulos Controlados y Aislados:** Basados en *EmotionPrompt*, en lugar de modificar las preguntas de MMLU, anexamos estímulos emocionales estandarizados y fijos (basados en teorías de Psicología como la *Teoría Cognitiva Social* y el *Auto-Monitoreo*) al final del prompt. Esto permite aislar la emoción como una variable independiente pura, eliminando variables de confusión relacionadas con el cambio de formato o vocabulario.
3. **Reproducibilidad:** El uso de un conjunto estático de frases asegura que el experimento sea 100% determinista y fácilmente reproducible por la comunidad científica.

## Estímulos Utilizados

Basado en la literatura, se inyectan las siguientes frases al final de las preguntas de MMLU (después de mostrar las opciones):

*   **Original (Baseline):** *(Sin texto extra)*
*   **Cortesía:** *"Could you please provide the correct answer? I would greatly appreciate your help."*
*   **Optimismo (Confianza):** *"Take a deep breath and think step by step. I believe in your abilities, you are an expert."*
*   **Ansiedad (Presión):** *"This is very important to my career. You'd better be sure of your answer or there will be severe consequences."*
*   **Enojo (Hostilidad):** *"Just give me the right answer immediately! I am tired of your mistakes!"*

## Requisitos Previos e Instalación

1. **Instalar dependencias de Python:**
   Asegúrate de tener instalado Python y ejecutar el siguiente comando en la raíz de "Estandar de LLM" para instalar las librerías necesarias:
   ```bash
   pip install -r requirements.txt
   ```

2. **Instalar y descargar modelos locales (Ollama):**
   Los scripts están configurados para ejecutarse usando modelos locales a través de [Ollama](https://ollama.com/). Primero, descarga los modelos que vas a evaluar:
   ```bash
   ollama pull nemotron3:33b
   ollama pull gemma4:31b
   ```
   *(Nota: Asegúrate de tener Ollama instalado y en ejecución en tu sistema)*

3. **Configurar la API (Solo para MT-Bench):**
   Para evaluar las respuestas abiertas de MT-Bench, se utiliza ChatGPT como Juez. Debes asegurarte de que el archivo `.env` en la carpeta `Estandar de LLM` tenga tu clave de OpenAI:
   ```env
   OPENAI_API_KEY=sk-tu-clave-real-aqui
   JUDGE_MODEL=gpt-4o
   ```

## Uso de los Scripts

Los scripts de evaluación están separados por el benchmark a evaluar (MMLU, GSM8K, MT-Bench). Debes navegar a la carpeta correspondiente para ejecutarlos.

### 1. Evaluar MMLU (Opción Múltiple)
Desde la carpeta `Estandar de LLM`, navega a la carpeta de MMLU:
```bash
cd "scripts_llm/evalucion_MMLU"
```
Ejecuta la evaluación del modelo que desees (esto generará una carpeta `results_nemotron` o `results_gemma`):
```bash
python evaluator_mmlu_nemotron.py
# o
python evaluator_mmlu_gemma.py
```

### 2. Evaluar GSM8K (Matemáticas)
Navega a la carpeta de GSM8K:
```bash
cd "scripts_llm/evaluator_GSM8K"
```
Ejecuta la evaluación:
```bash
python evaluator_gsm8k_nemotron.py
# o
python evaluator_gsm8k_gemma.py
```

### 3. Evaluar MT-Bench (Preguntas Abiertas)
Navega a la carpeta de MT-Bench:
```bash
cd "scripts_llm/evalutator_MT-Bench"
```
Ejecuta la evaluación para generar las respuestas:
```bash
python evaluator_mtbench_nemotron.py
# o
python evaluator_mtbench_gemma.py
```

## Análisis de Métricas

Una vez que tengas los resultados generados, puedes analizarlos. Ve a la carpeta donde corriste el evaluador y ejecuta el script de análisis indicando la carpeta de resultados.

Por ejemplo, para MMLU o GSM8K:
```bash
python analyze_metrics.py results_nemotron
# o
python analyze_metrics.py results_gemma
```

Para MT-Bench (que utiliza ChatGPT como juez y tomará un momento en analizar las preguntas):
```bash
python analyze_metrics_mtbench.py results_nemotron
```

El análisis imprimirá la precisión (Accuracy) o puntuación promedio en la consola y generará gráficos comparativos `.png` para visualizar el rendimiento del modelo bajo cada emoción.

## Resultados del Experimento (Para el Paper)

Hemos realizado evaluaciones exhaustivas utilizando el benchmark **MMLU** bajo los 5 estímulos del paradigma **EmotionPrompt** para dos modelos representativos de distintas arquitecturas y escalas de parámetros:
1. **Nemotron-3 33B** (33 mil millones de parámetros, arquitectura Transformer).
2. **Falcon Mamba 7B** (7 mil millones de parámetros, arquitectura de Espacio de Estados / State Space Model).

### Tabla de Precisión General (Accuracy %)

| Estímulo Emocional | Nemotron-3 33B (Transformer) | Falcon Mamba 7B (SSM) |
| :--- | :---: | :---: |
| **Original (Neutral - Baseline)** | 68.00% | **48.00%** |
| **Cortesía (Courtesy)** | 74.00% | 45.00% |
| **Optimismo (Confianza)** | 71.00% | 47.00% |
| **Ansiedad (Presión)** | **80.00%** | 47.00% |
| **Enojo (Hostilidad)** | 73.00% | 47.00% |

### Análisis Metodológico y Conclusiones

*   **Transformers (Nemotron-3 33B):** El modelo basado en la arquitectura tradicional de Transformer muestra una sensibilidad positiva muy alta a los estímulos emocionales, alcanzando su rendimiento óptimo bajo el prompt de **Ansiedad (80.00% vs. 68.00% en Baseline)**, lo que representa una mejora del **+12.00%**. Esto respalda fuertemente la hipótesis del paper *EmotionPrompt* de que los estímulos de presión social/profesional pueden forzar una mejor distribución de la atención en Transformers.
*   **State Space Models (Falcon Mamba 7B):** La arquitectura SSM/Mamba responde de forma opuesta o neutra ante los estímulos emocionales. Su precisión más alta se mantiene en el prompt **Original (48.00%)**, mientras que los estímulos emocionales degradan el rendimiento (hasta **45.00%** en Cortesía) o lo mantienen casi igual (**47.00%**).
*   **Implicaciones Teóricas:** Los Transformers pueden ponderar dinámicamente y contextualizar las frases emocionales mediante la auto-atención global sin perder el foco en la pregunta. Sin embargo, en arquitecturas SSM como Mamba, que comprimen la información de forma secuencial en un estado oculto de tamaño fijo, la adición de texto emocional no relacionado con el problema matemático o conceptual actúa como **ruido de compresión**, diluyendo la señal de la pregunta original y afectando negativamente la calidad de la respuesta.

### Visualizaciones y Notebooks de Análisis
Los gráficos comparativos generados por los scripts de análisis se guardan localmente como:
*   `accuracy_comparison_data_nemotron.png` (Nemotron)
*   `accuracy_comparison_data_falcon_mamba_results_falcon_mamba.png` (Falcon Mamba)

Para explorar el análisis interactivo de errores y distribuciones de respuestas por letra, abre los notebooks correspondientes en el entorno de desarrollo:
*   [analisis_nemotron.ipynb](file:///c:/Users/jonat/Desktop/ESCUELA/Articulo%20cientifico%20emociones/Estandar%20de%20LLM/analisis_nemotron.ipynb)
*   [analisis_falcon_mamba.ipynb](file:///c:/Users/jonat/Desktop/ESCUELA/Articulo%20cientifico%20emociones/Estandar%20de%20LLM/analisis_falcon_mamba.ipynb)

