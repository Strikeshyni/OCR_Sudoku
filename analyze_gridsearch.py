#!/usr/bin/env python3
"""
Script d'analyse des résultats du Grid Search
Génère des graphiques et tableaux comparatifs
"""

import pandas as pd
import sys
import os

def load_results(csv_path):
    """Charge les résultats du grid search"""
    if not os.path.exists(csv_path):
        print(f"❌ Fichier non trouvé: {csv_path}")
        print("Lancez d'abord: make gridsearch")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"✓ {len(df)} configurations chargées\n")
    return df

def print_summary(df):
    """Affiche un résumé statistique"""
    print("=" * 80)
    print("STATISTIQUES GLOBALES")
    print("=" * 80)
    print(f"Accuracy    : min={df['accuracy'].min():.4f}, max={df['accuracy'].max():.4f}, mean={df['accuracy'].mean():.4f}")
    print(f"F1-Score    : min={df['avg_f1_score'].min():.4f}, max={df['avg_f1_score'].max():.4f}, mean={df['avg_f1_score'].mean():.4f}")
    print(f"Training time: min={df['training_time'].min():.1f}s, max={df['training_time'].max():.1f}s, mean={df['training_time'].mean():.1f}s")
    print()

def print_top_configs(df, n=10):
    """Affiche les meilleures configurations"""
    print("=" * 80)
    print(f"TOP {n} CONFIGURATIONS (par F1-Score)")
    print("=" * 80)
    
    # Trier par F1-score décroissant
    top = df.nlargest(n, 'avg_f1_score')
    
    print(f"{'Rank':<6} {'Epochs':<8} {'Batch':<7} {'LR':<8} {'Momentum':<10} {'Accuracy':<10} {'F1-Score':<10} {'Time(min)':<10}")
    print("-" * 80)
    
    for idx, (i, row) in enumerate(top.iterrows(), 1):
        print(f"{idx:<6} {row['epochs']:<8} {row['batch_size']:<7} {row['learning_rate']:<8.4f} "
              f"{row['momentum']:<10.2f} {row['accuracy']:<10.4f} {row['avg_f1_score']:<10.4f} "
              f"{row['training_time']/60:<10.2f}")
    print()

def analyze_hyperparameters(df):
    """Analyse l'impact de chaque hyperparamètre"""
    print("=" * 80)
    print("IMPACT DES HYPERPARAMÈTRES")
    print("=" * 80)
    
    # Impact des epochs
    print("\n📊 EPOCHS:")
    for epoch in sorted(df['epochs'].unique()):
        subset = df[df['epochs'] == epoch]
        print(f"  {epoch:3d} epochs: F1={subset['avg_f1_score'].mean():.4f} ± {subset['avg_f1_score'].std():.4f}")
    
    # Impact du batch size
    print("\n📊 BATCH SIZE:")
    for batch in sorted(df['batch_size'].unique()):
        subset = df[df['batch_size'] == batch]
        print(f"  Batch {batch:3d}: F1={subset['avg_f1_score'].mean():.4f} ± {subset['avg_f1_score'].std():.4f}")
    
    # Impact du learning rate
    print("\n📊 LEARNING RATE:")
    for lr in sorted(df['learning_rate'].unique()):
        subset = df[df['learning_rate'] == lr]
        print(f"  LR {lr:.4f}: F1={subset['avg_f1_score'].mean():.4f} ± {subset['avg_f1_score'].std():.4f}")
    
    # Impact du momentum
    print("\n📊 MOMENTUM:")
    for mom in sorted(df['momentum'].unique()):
        subset = df[df['momentum'] == mom]
        print(f"  Momentum {mom:.2f}: F1={subset['avg_f1_score'].mean():.4f} ± {subset['avg_f1_score'].std():.4f}")
    print()

def find_pareto_frontier(df):
    """Trouve les configurations Pareto-optimales (F1 vs temps)"""
    print("=" * 80)
    print("CONFIGURATIONS PARETO-OPTIMALES (F1-Score vs Temps)")
    print("=" * 80)
    print("Ces configs offrent le meilleur compromis performance/temps\n")
    
    # Trier par temps d'entraînement
    df_sorted = df.sort_values('training_time')
    
    pareto = []
    max_f1 = -1
    
    for _, row in df_sorted.iterrows():
        if row['avg_f1_score'] > max_f1:
            pareto.append(row)
            max_f1 = row['avg_f1_score']
    
    print(f"{'Epochs':<8} {'Batch':<7} {'LR':<8} {'Momentum':<10} {'F1-Score':<10} {'Time(min)':<10} {'Speedup'}")
    print("-" * 80)
    
    baseline_time = pareto[-1]['training_time'] if pareto else 0
    
    for row in pareto:
        speedup = baseline_time / row['training_time'] if row['training_time'] > 0 else 0
        print(f"{row['epochs']:<8} {row['batch_size']:<7} {row['learning_rate']:<8.4f} "
              f"{row['momentum']:<10.2f} {row['avg_f1_score']:<10.4f} "
              f"{row['training_time']/60:<10.2f} {speedup:.2f}x")
    print()

def export_latex_table(df, output_file='results_table.tex'):
    """Exporte le top 5 en format LaTeX pour rapport"""
    top5 = df.nlargest(5, 'avg_f1_score')
    
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Top 5 configurations du Grid Search}\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("Rank & Epochs & Batch & LR & Momentum & Accuracy & F1-Score \\\\\n")
        f.write("\\hline\n")
        
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            f.write(f"{idx} & {row['epochs']} & {row['batch_size']} & "
                   f"{row['learning_rate']:.4f} & {row['momentum']:.2f} & "
                   f"{row['accuracy']:.4f} & {row['avg_f1_score']:.4f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ Tableau LaTeX exporté: {output_file}")

def main():
    if len(sys.argv) < 2:
        csv_path = "models/grid_search_results.csv"
    else:
        csv_path = sys.argv[1]
    
    print("\n" + "=" * 80)
    print("ANALYSE DES RÉSULTATS DU GRID SEARCH")
    print("=" * 80 + "\n")
    
    df = load_results(csv_path)
    
    print_summary(df)
    print_top_configs(df, n=10)
    analyze_hyperparameters(df)
    find_pareto_frontier(df)
    
    # Export optionnel
    export_latex_table(df)
    
    print("\n" + "=" * 80)
    print("RECOMMANDATIONS")
    print("=" * 80)
    
    best = df.nlargest(1, 'avg_f1_score').iloc[0]
    fastest_good = df[df['avg_f1_score'] > 0.975].nsmallest(1, 'training_time')
    
    print(f"\n🏆 Meilleure performance:")
    print(f"   Epochs={best['epochs']}, Batch={best['batch_size']}, "
          f"LR={best['learning_rate']:.4f}, Momentum={best['momentum']:.2f}")
    print(f"   F1={best['avg_f1_score']:.4f}, Temps={best['training_time']/60:.1f} min")
    
    if not fastest_good.empty:
        fast = fastest_good.iloc[0]
        print(f"\n⚡ Plus rapide avec >97.5% F1:")
        print(f"   Epochs={fast['epochs']}, Batch={fast['batch_size']}, "
              f"LR={fast['learning_rate']:.4f}, Momentum={fast['momentum']:.2f}")
        print(f"   F1={fast['avg_f1_score']:.4f}, Temps={fast['training_time']/60:.1f} min")
        
        speedup = best['training_time'] / fast['training_time']
        f1_diff = best['avg_f1_score'] - fast['avg_f1_score']
        print(f"   Gain: {speedup:.1f}x plus rapide, seulement -{f1_diff:.4f} F1-Score")
    
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
