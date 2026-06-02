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

## Uso de los Scripts

1.  **Evaluación (Inferencia):**
    `python evaluator.py`
    (Descarga MMLU en memoria, anexa los estímulos emocionales, envía los prompts al LLM configurado en la API OpenAI/NVIDIA NIM y guarda las respuestas en la carpeta `results`).

2.  **Análisis de Métricas:**
    `python analyze_metrics.py`
    (Calcula el *Accuracy* para cada emoción y genera el gráfico `accuracy_comparison.png`).
