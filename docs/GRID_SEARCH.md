# Grid Search pour Optimisation CNN

## Vue d'ensemble

Le système de grid search permet d'optimiser automatiquement les hyperparamètres du CNN pour obtenir les meilleures performances sur la reconnaissance de chiffres MNIST.

## Fonctionnalités

### Métriques calculées

Pour chaque configuration testée, le système calcule :

- **Accuracy globale** : Pourcentage de prédictions correctes
- **F1-Score moyen** : Moyenne harmonique de précision et recall sur les 10 classes
- **Précision par classe** : Pour chaque chiffre 0-9
- **Recall par classe** : Pour chaque chiffre 0-9
- **F1-Score par classe** : Pour chaque chiffre 0-9
- **Matrice de confusion** : 10×10 pour analyse détaillée
- **Temps d'entraînement** : Durée en minutes

### Hyperparamètres optimisés

- **Epochs** : Nombre d'itérations sur le dataset (10, 15, 20)
- **Batch size** : Taille des mini-batches (32, 64)
- **Learning rate** : Taux d'apprentissage (0.005, 0.01, 0.02)
- **Momentum** : Inertie du gradient (0.0, 0.9)

**Total : 36 configurations** (3×2×3×2)

## Utilisation

### Option 1 : Grid search complet

```bash
make gridsearch
```

Cela va :
1. Compiler le programme de grid search
2. Tester toutes les configurations (36 configs × 10-20 époques)
3. Générer les fichiers de résultats

**⏱️ Durée estimée** : 2-4 heures selon votre CPU

### Option 2 : Training avec meilleurs paramètres

Après avoir exécuté le grid search, vous pouvez utiliser automatiquement les meilleurs paramètres :

```bash
make train
```

Le programme `train_cnn` détecte automatiquement `models/best_params.txt` et utilise les paramètres optimisés.

## Fichiers générés

### `models/grid_search_results.csv`

Tableau complet de tous les résultats au format CSV :

```csv
epochs,batch_size,learning_rate,momentum,accuracy,avg_f1_score,training_time,precision_0,recall_0,f1_0,...
20,32,0.01,0.9,0.9845,0.9843,25.3,0.987,0.991,0.989,...
15,64,0.01,0.0,0.9821,0.9819,18.7,0.985,0.988,0.986,...
```

### `models/best_params.txt`

Meilleurs hyperparamètres trouvés :

```
# Meilleurs hyperparamètres trouvés par Grid Search
EPOCHS=20
BATCH_SIZE=32
LEARNING_RATE=0.0100
MOMENTUM=0.90

# Métriques obtenues
ACCURACY=0.9845
AVG_F1_SCORE=0.9843
TRAINING_TIME_MIN=25.30
```

### `models/cnn_weights_optimized.bin`

Modèle CNN entraîné avec les meilleurs paramètres, prêt à l'emploi.

## Affichage des résultats

Le programme affiche :

### TOP 5 des configurations

```
┌──────┬────────┬───────┬──────┬──────────┬──────────┬──────────┬──────────┐
│ Rank │ Epochs │ Batch │  LR  │ Momentum │ Accuracy │   F1     │ Time(min)│
├──────┼────────┼───────┼──────┼──────────┼──────────┼──────────┼──────────┤
│   1  │   20   │  32   │0.0100│   0.90   │  0.9845  │  0.9843  │  25.30   │
│   2  │   15   │  64   │0.0100│   0.00   │  0.9821  │  0.9819  │  18.70   │
│   3  │   20   │  64   │0.0050│   0.90   │  0.9815  │  0.9813  │  24.50   │
│   4  │   15   │  32   │0.0200│   0.90   │  0.9809  │  0.9807  │  19.20   │
│   5  │   10   │  32   │0.0100│   0.00   │  0.9795  │  0.9793  │  12.80   │
└──────┴────────┴───────┴──────┴──────────┴──────────┴──────────┴──────────┘
```

### Métriques détaillées par classe

Pour la meilleure configuration :

```
┌───────┬───────────┬──────────┬──────────┐
│ Digit │ Precision │  Recall  │ F1-Score │
├───────┼───────────┼──────────┼──────────┤
│   0   │  0.9872   │  0.9910  │  0.9891  │
│   1   │  0.9905   │  0.9935  │  0.9920  │
│   2   │  0.9823   │  0.9801  │  0.9812  │
│   3   │  0.9845   │  0.9821  │  0.9833  │
│   4   │  0.9801   │  0.9834  │  0.9817  │
│   5   │  0.9789   │  0.9765  │  0.9777  │
│   6   │  0.9856   │  0.9878  │  0.9867  │
│   7   │  0.9834   │  0.9812  │  0.9823  │
│   8   │  0.9812   │  0.9789  │  0.9800  │
│   9   │  0.9798   │  0.9823  │  0.9810  │
└───────┴───────────┴──────────┴──────────┘
```

## Interprétation des métriques

### Accuracy vs F1-Score

- **Accuracy** : Simple pourcentage de prédictions correctes
  - Peut être trompeur si les classes sont déséquilibrées
  
- **F1-Score** : Moyenne harmonique de précision et recall
  - Meilleure métrique pour évaluer la qualité globale
  - Tient compte des faux positifs ET faux négatifs

### Précision vs Recall

- **Précision** : "Quand je prédis un 5, est-ce vraiment un 5 ?"
  - `TP / (TP + FP)`
  - Important pour éviter les fausses alarmes
  
- **Recall** : "Est-ce que je détecte tous les 5 ?"
  - `TP / (TP + FN)`
  - Important pour ne rien manquer

### F1-Score

- **F1 = 2 × (Précision × Recall) / (Précision + Recall)**
- Équilibre entre précision et recall
- **0.98+ = Excellent**, 0.95-0.98 = Très bon, 0.90-0.95 = Bon

## Analyse post-grid search

### 1. Charger les résultats dans Excel/Python

```python
import pandas as pd
df = pd.read_csv('models/grid_search_results.csv')
df = df.sort_values('avg_f1_score', ascending=False)
print(df.head(10))
```

### 2. Visualiser les tendances

- Impact du learning rate sur la convergence
- Trade-off temps vs accuracy
- Effet du momentum sur la stabilité

### 3. Tester les top configs

Vous pouvez manuellement éditer `models/best_params.txt` pour tester différentes configurations du top 5.

## Performance attendue

### Objectifs

- **Accuracy** : > 98%
- **F1-Score moyen** : > 0.98
- **F1 par classe** : > 0.97 pour chaque chiffre

### Temps typiques

Sur un CPU moderne (i7/i9 ou Ryzen 7/9) :

- 1 config (10 epochs) : ~8-12 minutes
- 1 config (20 epochs) : ~15-25 minutes
- Grid search complet (36 configs) : **2-4 heures**

## Conseils

1. **Lancez le grid search la nuit** ou pendant une période où vous n'utilisez pas votre PC
2. **Surveillez la température CPU** pour éviter la surchauffe
3. **Sauvegardez régulièrement** les fichiers .csv générés
4. **Analysez les résultats** avant de relancer avec une grille plus fine

## Troubleshooting

### Le programme s'arrête

- Vérifiez l'espace disque disponible
- Vérifiez la RAM disponible (~2-3 GB nécessaires)
- Consultez les logs pour voir quelle config a échoué

### Résultats incohérents

- Fixez la seed aléatoire dans le code
- Augmentez le nombre d'époques
- Vérifiez que MNIST est bien chargé

### Pas d'amélioration

Si toutes les configs donnent ~98% :
- Le modèle est déjà optimal pour MNIST
- Testez sur vos images de Sudoku réelles
- Ajustez le preprocessing si nécessaire
