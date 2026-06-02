import os
import json
import csv
import subprocess
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Estímulos validados basados en el paper "EmotionPrompt" (Li et al., 2023)
EMOTION_STIMULI = {
    "original": "",
    "courtesy": "Could you please provide the correct answer? I would greatly appreciate your help.",
    "optimism": "Take a deep breath and think step by step. I believe in your abilities, you are an expert.",
    "anxiety": "This is very important to my career. You'd better be sure of your answer or there will be severe consequences.",
    "anger": "Just give me the right answer immediately! I am tired of your mistakes!"
}

def format_prompt(emotion_stimulus, question, choices):
    options_text = f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
    
    # Metodología EmotionPrompt: anexar el estímulo al final
    if emotion_stimulus:
        return (
            f"Question: {question}\n\n"
            f"Options:\n{options_text}\n\n"
            f"{emotion_stimulus}\n"
            f"Answer:"
        )
    return (
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Answer:"
    )

def extract_answer(text):
    text = text.strip().upper()
    if not text:
        return -1
    
    # Heurística simple para extraer la letra de la opción (A, B, C, D)
    if 'A' in text[:5]: return 0
    if 'B' in text[:5]: return 1
    if 'C' in text[:5]: return 2
    if 'D' in text[:5]: return 3
    
    return -1

def main():

    # Nombre del modelo local 
    model_name = os.getenv("MODEL_NAME", "gemma4:31b")
    print(f"Usando el modelo local: {model_name}")
    
    print("Loading MMLU dataset (cais/mmlu)...")
    # Descargamos el benchmark directamente en memoria
    dataset = load_dataset("cais/mmlu", "all", split="test")
    
    # Muestreamos 100 preguntas de forma determinista
    print("Sampling 100 questions for the experiment...")
    dataset = dataset.shuffle(seed=42).select(range(100))
    
    output_dir = "results_gemma"
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluamos cada emoción
    for emotion_name, stimulus in EMOTION_STIMULI.items():
        output_file = os.path.join(output_dir, f"results_{emotion_name}.csv")
        print(f"\nEvaluating MMLU with '{emotion_name}' prompt...")
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ["question", "prompt_used", "target_answer", "predicted_answer", "is_correct", "subject", "emotion"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in tqdm(dataset, desc=emotion_name):
                prompt = format_prompt(stimulus, item['question'], item['choices'])
                
                try:
                    # Ejecutar modelo mediante comando de terminal
                    command = ["ollama", "run", model_name, prompt]
                    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
                    
                    if result.returncode != 0:
                        print(f"Error corriendo modelo: {result.stderr}")
                        model_answer = ""
                    else:
                        model_answer = result.stdout.strip()
                    
                    predicted_idx = extract_answer(model_answer)
                    is_correct = (predicted_idx == item['answer'])
                    
                    record = {
                        "question": item['question'],
                        "prompt_used": prompt,
                        "target_answer": item['answer'],
                        "predicted_answer": model_answer,
                        "is_correct": is_correct,
                        "subject": item['subject'],
                        "emotion": emotion_name
                    }
                    writer.writerow(record)
                except Exception as e:
                    print(f"Error querying API: {e}")

if __name__ == "__main__":
    main()
