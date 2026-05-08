import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ruta al dataset de Llama 3
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, "dataset_experimento", "llama3_intensidad_experimento.csv")

def leer_y_analizar_dataset():
    if not os.path.exists(dataset_path):
        print(f"Error: No se encontró el dataset en {dataset_path}")
        return

    print("=== Leyendo Dataset de Intensidad Emocional (Llama 3) ===\n")
    df = pd.read_csv(dataset_path)

    # Mostrar información general
    print("1. Información del Dataset:")
    print(df.info())
    print("\n2. Primeras filas:")
    print(df.head())

    # Estadísticas descriptivas de longitud y diversidad por nivel de intensidad
    print("\n3. Estadísticas por nivel de intensidad:")
    estadisticas = df.groupby('intensidad')[['longitud', 'diversidad']].mean().reset_index()
    print(estadisticas)

    # Opcional: Generar gráficos
    print("\nGenerando gráficos...")
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 5))
    
    # Gráfico de Longitud
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x='intensidad', y='longitud', capsize=.1)
    plt.xticks(rotation=45, ha='right')
    plt.title('Longitud media de respuesta por Intensidad')
    plt.tight_layout()

    # Gráfico de Diversidad
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x='intensidad', y='diversidad', capsize=.1)
    plt.xticks(rotation=45, ha='right')
    plt.title('Diversidad léxica media por Intensidad')
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    leer_y_analizar_dataset()
