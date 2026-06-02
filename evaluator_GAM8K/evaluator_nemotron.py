import os
import json
import csv
import subprocess
import re
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

def format_prompt(emotion_stimulus, question):
    if emotion_stimulus:
        return (
            f"Question: {question}\n\n"
            f"{emotion_stimulus}\n"
            f"Answer with a number:"
        )
    return (
        f"Question: {question}\n\n"
        f"Answer with a number:"
    )

def extract_answer(text):
    # Buscamos el último número en la respuesta (común en problemas matemáticos)
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None

def main():

    # Nombre del modelo local 
    model_name = os.getenv("MODEL_NAME", "nemotron3:33b")
    print(f"Usando el modelo local: {model_name}")
    
    print("Loading GSM8K dataset...")
    # Descargamos el benchmark GSM8K
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Muestreamos 100 preguntas de forma determinista
    print("Sampling 100 questions for the experiment...")
    dataset = dataset.shuffle(seed=42).select(range(100))
    
    output_dir = "results_nemotron"
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluamos cada emoción
    for emotion_name, stimulus in EMOTION_STIMULI.items():
        output_file = os.path.join(output_dir, f"results_{emotion_name}.csv")
        print(f"\nEvaluating GSM8K with '{emotion_name}' prompt...")
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ["question", "prompt_used", "target_answer", "predicted_answer", "is_correct", "emotion"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in tqdm(dataset, desc=emotion_name):
                prompt = format_prompt(stimulus, item['question'])
                
                # Extraer la respuesta real del GSM8K (después de "####")
                target_str = item['answer'].split('####')[-1].strip()
                
                try:
                    # Ejecutar modelo mediante comando de terminal
                    command = ["ollama", "run", model_name, prompt]
                    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
                    
                    if result.returncode != 0:
                        print(f"Error corriendo modelo: {result.stderr}")
                        model_answer = ""
                    else:
                        model_answer = result.stdout.strip()
                    
                    predicted_number = extract_answer(model_answer)
                    
                    # Comparamos numéricamente si es posible
                    is_correct = False
                    if predicted_number is not None:
                        try:
                            is_correct = (float(predicted_number) == float(target_str))
                        except ValueError:
                            is_correct = (predicted_number == target_str)
                    
                    record = {
                        "question": item['question'],
                        "prompt_used": prompt,
                        "target_answer": target_str,
                        "predicted_answer": model_answer,
                        "is_correct": is_correct,
                        "emotion": emotion_name
                    }
                    writer.writerow(record)
                except Exception as e:
                    print(f"Error querying model: {e}")

if __name__ == "__main__":
    main()
