#!/bin/bash

# Script pour lancer un grid search rapide (test)
# Utilise seulement un sous-ensemble du dataset MNIST pour tester rapidement

echo "========================================="
echo "  GRID SEARCH RAPIDE (TEST)"
echo "========================================="
echo ""
echo "Ce script lance un grid search avec:"
echo "  - 3 configurations seulement"
echo "  - 5 époques maximum"
echo "  - Pour tester la fonctionnalité"
echo ""
echo "Pour un vrai grid search complet:"
echo "  make gridsearch"
echo ""

# Compiler
make clean
make gridsearch

# Lancer
./build/grid_search data/mnist models/

echo ""
echo "Résultats sauvegardés dans models/"
echo "  - grid_search_results.csv"
echo "  - best_params.txt"
echo "  - cnn_weights_optimized.bin"
