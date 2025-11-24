# Guide de Démarrage Rapide - OCR Sudoku Solver

## Installation et Compilation

### 1. Télécharger les dépendances

Les bibliothèques stb_image sont déjà incluses. Il ne reste qu'à télécharger le dataset MNIST :

```bash
./download_mnist.sh
```

Cela téléchargera environ 50 MB de données (60,000 images d'entraînement + 10,000 de test).

### 2. Compiler le projet

```bash
# Compiler le solveur Sudoku principal
make all

# Compiler le programme d'entraînement CNN
make train
```

## Entraînement du Modèle CNN

**IMPORTANT**: Avant d'utiliser le solveur, vous devez entraîner le modèle de reconnaissance de chiffres.

```bash
# Lancer l'entraînement (peut prendre 2-8 heures sur CPU)
./build/train_cnn data/mnist models/cnn_weights.bin
```

**Paramètres d'entraînement** (modifiables dans `src/train_cnn.c`):
- Époques: 10 par défaut (augmenter à 20-30 pour ~98% de précision)
- Batch size: 32
- Learning rate: 0.01

**Note**: L'entraînement sur CPU sera lent. Voici les temps estimés:
- CPU moderne (8 cores): ~4-6 heures pour 10 époques
- CPU ancien (2-4 cores): ~8-12 heures pour 10 époques

Pour accélérer l'entraînement, vous pouvez réduire le nombre d'époques à 5, mais la précision sera moindre (~92-94% au lieu de ~96-97%).

## Utilisation

### Résoudre une grille de Sudoku

```bash
./build/sudoku_solver input_sudoku.jpg output_solved.png
```

Le programme effectue automatiquement:
1. ✓ Prétraitement de l'image (niveaux de gris, binarisation, débruitage)
2. ✓ Détection de la grille (transformée de Hough, contours)
3. ✓ Extraction et redressement de la grille
4. ✓ Découpage en 81 cases (9×9)
5. ✓ Reconnaissance des chiffres via CNN
6. ✓ Résolution du Sudoku (backtracking MRV)
7. ✓ Composition de l'image finale avec chiffres complétés (en rouge)

### Fichiers de debug générés

Pendant l'exécution, plusieurs images intermédiaires sont sauvegardées:
- `debug_edges.png`: Détection des contours
- `debug_grid.png`: Grille extraite et redressée
- `debug_cell_X_Y.png`: Exemples de cases découpées

## Format d'image requis

**Images acceptées**:
- Format: JPG, PNG, BMP
- Résolution: minimum 800×800 pixels recommandé
- Qualité: Bonne luminosité, contraste net entre lignes et fond
- Orientation: La grille doit être visible entièrement

**Conseils pour de meilleurs résultats**:
- Prenez la photo avec un bon éclairage
- Évitez les ombres fortes
- Cadrez la grille au centre
- Utilisez une grille imprimée plutôt que manuscrite (pour la version actuelle)

## Architecture du Projet

```
OCR_Sudoku/
├── src/
│   ├── main.c                  # Pipeline principal
│   ├── train_cnn.c             # Entraînement CNN
│   ├── utils.c/.h              # Utilitaires (matrices, math)
│   ├── image_loader.c/.h       # Chargement images (stb)
│   ├── preprocessor.c/.h       # Prétraitement (Otsu, Gaussian, Sobel)
│   ├── grid_detector.c/.h      # Détection grille (Hough, contours)
│   ├── perspective.c/.h        # Transformation perspective
│   ├── cell_extractor.c/.h     # Extraction cases 9×9
│   ├── cnn_model.c/.h          # Architecture CNN (LeNet-5 style)
│   ├── cnn_training.c/.h       # Backpropagation, optimiseur SGD
│   ├── dataset_loader.c/.h     # Chargement MNIST
│   ├── sudoku_solver.c/.h      # Backtracking MRV
│   └── image_composer.c/.h     # Composition image finale
│
├── models/
│   └── cnn_weights.bin         # Poids du CNN entraîné
│
├── data/
│   ├── mnist/                  # Dataset MNIST (60k train + 10k test)
│   └── test_images/            # Vos images de test
│
├── Makefile                    # Compilation
├── download_mnist.sh           # Script téléchargement MNIST
└── README.md                   # Documentation
```

## Modèle CNN (Architecture LeNet-5 adaptée)

```
Input: 28×28×1 (image en niveaux de gris)
  ↓
Conv2D: 6 filtres 5×5 + ReLU → 24×24×6
  ↓
MaxPool: 2×2 → 12×12×6
  ↓
Conv2D: 16 filtres 5×5 + ReLU → 8×8×16
  ↓
MaxPool: 2×2 → 4×4×16 = 256 neurones
  ↓
Dense: 120 neurones + ReLU
  ↓
Dense: 10 neurones (classes 0-9)
  ↓
Softmax
```

**Total de paramètres**: ~30,000

## Performances attendues

**Reconnaissance de chiffres (CNN)**:
- Précision MNIST: 96-98% (après 20-30 époques)
- Temps d'inférence: <5ms par chiffre (CPU)

**Résolution Sudoku**:
- Temps de résolution: <100ms par grille (backtracking MRV)
- Taux de succès: >95% sur grilles imprimées claires

**Pipeline complet**:
- Temps total: 2-5 secondes par image (selon résolution)

## Limitations et améliorations possibles

**Limitations actuelles**:
- Chiffres manuscrits peuvent avoir une précision réduite (MNIST est principalement manuscrit mais standardisé)
- Grilles floues ou avec mauvais éclairage peuvent échouer la détection
- Transformation perspective inverse non complètement implémentée (overlay sur grille normalisée)

**Améliorations futures**:
- Augmentation de données plus aggressive (rotation, bruit, distorsions)
- Architecture CNN plus profonde (ResNet-style)
- Entraînement sur dataset de chiffres imprimés spécifiques au Sudoku
- Transformation perspective inverse complète pour overlay sur image originale
- Support de grilles manuscrites avec transfer learning
- Optimisations SIMD pour accélérer les convolutions

## Dépannage

### Erreur: "Impossible de détecter la grille de Sudoku"

**Causes possibles**:
- Image trop sombre ou surexposée
- Grille incomplète ou coupée
- Lignes trop fines ou effacées

**Solutions**:
- Améliorer l'éclairage et reprendre la photo
- Augmenter le contraste de l'image en preprocessing
- Vérifier que toutes les lignes de la grille sont visibles

### Erreur: "Impossible de résoudre la grille"

**Causes possibles**:
- Erreurs de reconnaissance OCR (chiffres mal reconnus)
- Grille invalide (plusieurs solutions ou aucune solution)

**Solutions**:
- Vérifier les fichiers `debug_cell_*.png` pour voir les chiffres mal reconnus
- Comparer la grille reconnue (affichée dans le terminal) avec l'originale
- Réentraîner le modèle sur plus d'époques

### Erreur: "Échec du chargement des poids CNN"

**Cause**: Le modèle n'a pas encore été entraîné

**Solution**:
```bash
./build/train_cnn data/mnist models/cnn_weights.bin
```

## Licence

MIT License - Projet éducatif

## Auteur

Projet personnel de démonstration d'un pipeline complet OCR + résolution algorithmique en C pur, sans dépendances lourdes (OpenCV, TensorFlow, etc.).
