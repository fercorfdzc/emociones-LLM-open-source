import os
import json
import csv
import subprocess
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

EMOTION_STIMULI = {
    "original": "",
    "courtesy": "Could you please provide the correct answer? I would greatly appreciate your help.",
    "optimism": "Take a deep breath and think step by step. I believe in your abilities, you are an expert.",
    "anxiety": "This is very important to my career. You'd better be sure of your answer or there will be severe consequences.",
    "anger": "Just give me the right answer immediately! I am tired of your mistakes!"
}

def format_prompt(emotion_stimulus, question):
    if emotion_stimulus:
        return f"{question}\n\n{emotion_stimulus}"
    return question

def main():
    model_name = os.getenv("MODEL_NAME", "nemotron3:33b")
    print(f"Usando el modelo local: {model_name}")
    
    print("Loading MT-Bench dataset...")
    dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    
    output_dir = "results_nemotron"
    os.makedirs(output_dir, exist_ok=True)
    
    for emotion_name, stimulus in EMOTION_STIMULI.items():
        output_file = os.path.join(output_dir, f"results_{emotion_name}.csv")
        print(f"\nEvaluating MT-Bench with '{emotion_name}' prompt...")
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ["prompt_id", "category", "question", "prompt_used", "predicted_answer", "emotion"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in tqdm(dataset, desc=emotion_name):
                # Solo evaluamos el primer turno
                question = item['prompt'][0]
                prompt = format_prompt(stimulus, question)
                
                try:
                    command = ["ollama", "run", model_name, prompt]
                    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
                    
                    if result.returncode != 0:
                        print(f"Error corriendo modelo: {result.stderr}")
                        model_answer = ""
                    else:
                        model_answer = result.stdout.strip()
                    
                    record = {
                        "prompt_id": item['prompt_id'],
                        "category": item['category'],
                        "question": question,
                        "prompt_used": prompt,
                        "predicted_answer": model_answer,
                        "emotion": emotion_name
                    }
                    writer.writerow(record)
                except Exception as e:
                    print(f"Error querying model: {e}")

if __name__ == "__main__":
    main()
