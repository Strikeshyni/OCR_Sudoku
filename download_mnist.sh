#!/bin/bash

# Script de téléchargement du dataset MNIST

MNIST_DIR="data/mnist"

echo "========================================="
echo "  TÉLÉCHARGEMENT DU DATASET MNIST"
echo "========================================="
echo ""

# Créer le dossier si nécessaire
mkdir -p "$MNIST_DIR"

cd "$MNIST_DIR" || exit 1

# URLs des fichiers MNIST
TRAIN_IMAGES="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
TRAIN_LABELS="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
TEST_IMAGES="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
TEST_LABELS="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

# Télécharger les fichiers
echo "Téléchargement des fichiers MNIST..."

if [ ! -f "train-images-idx3-ubyte" ]; then
    echo "  - Images d'entraînement..."
    wget -q --show-progress "$TRAIN_IMAGES"
    gunzip train-images-idx3-ubyte.gz
fi

if [ ! -f "train-labels-idx1-ubyte" ]; then
    echo "  - Labels d'entraînement..."
    wget -q --show-progress "$TRAIN_LABELS"
    gunzip train-labels-idx1-ubyte.gz
fi

if [ ! -f "t10k-images-idx3-ubyte" ]; then
    echo "  - Images de test..."
    wget -q --show-progress "$TEST_IMAGES"
    gunzip t10k-images-idx3-ubyte.gz
fi

if [ ! -f "t10k-labels-idx1-ubyte" ]; then
    echo "  - Labels de test..."
    wget -q --show-progress "$TEST_LABELS"
    gunzip t10k-labels-idx1-ubyte.gz
fi

echo ""
echo "✓ Dataset MNIST téléchargé avec succès!"
echo ""
echo "Fichiers présents dans $MNIST_DIR:"
ls -lh

echo ""
echo "Vous pouvez maintenant entraîner le modèle avec:"
echo "  make train"
