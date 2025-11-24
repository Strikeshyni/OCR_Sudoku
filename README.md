# OCR Sudoku Solver

Solveur de Sudoku complet en C pur avec reconnaissance optique de caractères (OCR) basée sur un réseau de neurones convolutif (CNN) implémenté from scratch.

## Fonctionnalités

- **Prétraitement d'image** : conversion en niveaux de gris, binarisation Otsu, débruitage
- **Détection de grille** : détection des lignes via transformée de Hough, extraction du quadrilatère principal
- **Extraction de cases** : découpage de la grille en 81 cases (9×9), normalisation 28×28 pixels
- **Reconnaissance de chiffres** : CNN implémenté en C avec entraînement complet (backpropagation, SGD/Adam)
- **Résolution** : algorithme de backtracking optimisé avec heuristique MRV
- **Reconstruction** : génération de l'image finale avec les chiffres complétés

## Architecture

```
OCR_Sudoku/
├── src/
│   ├── main.c                  # Pipeline principal
│   ├── train_cnn.c             # Programme d'entraînement CNN
│   ├── utils.c/.h              # Utilitaires (matrices, maths)
│   ├── image_loader.c/.h       # Chargement/sauvegarde images
│   ├── preprocessor.c/.h       # Prétraitement images
│   ├── grid_detector.c/.h      # Détection de grille
│   ├── perspective.c/.h        # Transformation perspective
│   ├── cell_extractor.c/.h     # Extraction des cases
│   ├── cnn_model.c/.h          # Architecture CNN (forward/inference)
│   ├── cnn_training.c/.h       # Backpropagation et optimiseur
│   ├── dataset_loader.c/.h     # Chargement MNIST/IDX
│   ├── sudoku_solver.c/.h      # Solveur backtracking
│   └── image_composer.c/.h     # Reconstruction image
├── models/
│   └── cnn_weights.bin         # Poids du CNN entraîné
├── data/
│   ├── mnist/                  # Dataset MNIST
│   └── test_images/            # Images de test
├── Makefile                    # Compilation
└── CMakeLists.txt              # Alternative CMake
```

## Compilation

```bash
# Avec Make
make all

# Entraînement du CNN (nécessite MNIST téléchargé)
make train

# Grid Search pour optimiser les hyperparamètres
make gridsearch

# Utilisation
./build/sudoku_solver input.jpg output.png
```

## Optimisation des Hyperparamètres

Le projet inclut un système de **grid search automatique** pour optimiser les performances du CNN :

```bash
make gridsearch
```

Cela va :
- Tester 36 configurations d'hyperparamètres (epochs, batch size, learning rate, momentum)
- Calculer les métriques détaillées (accuracy, F1-score, precision, recall par classe)
- Sauvegarder les résultats dans `models/grid_search_results.csv`
- Identifier et sauvegarder la meilleure configuration dans `models/best_params.txt`
- Entraîner le modèle final avec les meilleurs paramètres

**Durée estimée** : 2-4 heures selon votre CPU

Les métriques calculées incluent :
- **Accuracy globale** : % de prédictions correctes
- **F1-Score moyen** : métrique équilibrée entre précision et recall
- **Précision/Recall/F1 par classe** : analyse détaillée pour chaque chiffre 0-9
- **Matrice de confusion** : visualisation des erreurs
- **Temps d'entraînement** : optimisation du compromis vitesse/qualité

Voir [docs/GRID_SEARCH.md](docs/GRID_SEARCH.md) pour plus de détails.

## Dépendances

- **stb_image.h / stb_image_write.h** : chargement/sauvegarde d'images (header-only, inclus)
- **Compilateur C99** : GCC ou Clang
- **libm** : bibliothèque mathématique standard

Pas de dépendances externes lourdes (OpenCV, TensorFlow, etc.)

## Téléchargement du dataset MNIST

```bash
mkdir -p data/mnist
cd data/mnist
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

## Performance

- **Entraînement CNN** : ~30 époques sur MNIST (CPU, ~30min-2h selon machine)
- **Précision OCR attendue** : >98.3% sur chiffres manuscrits
- **Temps résolution** : <100ms par grille Sudoku

## Licence

MIT License - Projet personnel éducatif
