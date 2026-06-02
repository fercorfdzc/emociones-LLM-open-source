import os
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import re
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def get_judge_score(client, judge_model, question, answer):
    # Prompt típico de LLM-as-a-judge
    prompt = f"""
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question below. Your evaluation should consider helpfulness, relevance, accuracy, depth, creativity, and level of detail.
You will rate the response on a scale of 1 to 10. 
First, provide a short explanation of your evaluation. Then, you must output a single number indicating the score in the format "Score: <rating>".

[User Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]
"""
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        
        # Extraer el puntaje con regex
        match = re.search(r'Score:\s*([0-9]+(?:\.[0-9]+)?)', content, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        # Fallback simple
        numbers = re.findall(r'\b(?:10|[1-9])\b', content)
        if numbers:
            return float(numbers[-1])
            
        return 0.0
    except Exception as e:
        print(f"Error calling judge model: {e}")
        return 0.0

def main():
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "results_nemotron"
        
    print(f"Analizando resultados (MT-Bench) en la carpeta: {results_dir}")
    emotions = ["original", "courtesy", "optimism", "anxiety", "anger"]
    summary = {}
    
    # Configurar juez usando la API oficial de OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "local-key":
        print("ADVERTENCIA: Asegúrate de configurar tu OPENAI_API_KEY real en el archivo .env para usar ChatGPT.")
        
    # Por defecto usamos gpt-4o, o el que esté en el .env
    judge_model = os.getenv("JUDGE_MODEL", "gpt-4o") 
    
    print(f"Conectando a la API de OpenAI (Modelo juez: {judge_model})")
    client = OpenAI(api_key=api_key)
    
    # Carpeta donde se guardarán las evaluaciones del juez para no repetirlas
    scored_dir = "scored_results"
    os.makedirs(scored_dir, exist_ok=True)
    
    for emotion in emotions:
        results_file = os.path.join(results_dir, f"results_{emotion}.csv")
        scored_file = os.path.join(scored_dir, f"scored_{results_dir}_{emotion}.csv")
        
        if not os.path.exists(results_file):
            print(f"Results for {emotion} not found.")
            continue
            
        df = pd.read_csv(results_file)
        if df.empty:
            continue
            
        # Revisamos si ya lo evaluamos previamente
        if os.path.exists(scored_file):
            print(f"Cargando evaluaciones previas para '{emotion}'...")
            df = pd.read_csv(scored_file)
        else:
            print(f"El Juez ({judge_model}) está evaluando respuestas de '{emotion}'...")
            scores = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Juzgando"):
                score = get_judge_score(client, judge_model, row['question'], row['predicted_answer'])
                scores.append(score)
            
            df['score'] = scores
            df.to_csv(scored_file, index=False)
            
        avg_score = df['score'].mean()
        summary[emotion] = avg_score
            
    if not summary:
        print("No results to analyze.")
        return
        
    # Imprimir tabla
    print(f"\n=== RESULTADOS MT-BENCH (Promedio 1-10) ===")
    for emotion, score in summary.items():
        print(f"{emotion.capitalize():<10} : {score:.2f} / 10")
        
    # Graficar
    plt.figure(figsize=(8, 6))
    emotions_labels = [e.capitalize() for e in summary.keys()]
    scores_list = list(summary.values())
    
    bars = plt.bar(emotions_labels, scores_list, color=['gray', 'blue', 'green', 'orange', 'red'])
    plt.ylabel('Average Score (1-10)')
    plt.title(f'MT-Bench Score by Emotional Prompt ({results_dir})')
    plt.ylim(0, 10)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.2f}', ha='center', va='bottom')
        
    plt.tight_layout()
    plot_path = f"score_comparison_{results_dir}.png"
    plt.savefig(plot_path)
    print(f"\nSaved plot to {plot_path}")

if __name__ == "__main__":
    main()
