import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def analyze_results(results_file):
    try:
        df = pd.read_csv(results_file)
    except Exception as e:
        print(f"Error reading {results_file}: {e}")
        return 0
        
    if df.empty:
        return 0
    
    # Accuracy general
    overall_accuracy = df['is_correct'].mean() * 100
    return overall_accuracy

def main():
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "results_nemotron"
        
    print(f"Analizando resultados en la carpeta: {results_dir}")
    emotions = ["original", "courtesy", "optimism", "anxiety", "anger"]
    summary = {}
    
    for emotion in emotions:
        results_file = os.path.join(results_dir, f"results_{emotion}.csv")
        if os.path.exists(results_file):
            overall_acc = analyze_results(results_file)
            summary[emotion] = overall_acc
        else:
            print(f"Results for {emotion} not found.")
            
    if not summary:
        print("No results to analyze.")
        return
        
    # Imprimir tabla
    print("\n=== RESULTADOS GSM8K ===")
    for emotion, acc in summary.items():
        print(f"{emotion.capitalize():<10} : {acc:.2f}%")
        
    # Graficar
    plt.figure(figsize=(8, 6))
    emotions_labels = [e.capitalize() for e in summary.keys()]
    accuracies = list(summary.values())
    
    bars = plt.bar(emotions_labels, accuracies, color=['gray', 'blue', 'green', 'orange', 'red'])
    plt.ylabel('Accuracy (%)')
    plt.title(f'GSM8K Accuracy by Emotional Prompt ({results_dir})')
    plt.ylim(0, max(accuracies) + 10 if accuracies else 100)
    
    # Añadir valores a las barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')
        
    plt.tight_layout()
    plot_path = f"accuracy_comparison_{results_dir}.png"
    plt.savefig(plot_path)
    print(f"\nSaved plot to {plot_path}")

if __name__ == "__main__":
    main()
