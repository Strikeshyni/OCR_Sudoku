#ifndef CELL_EXTRACTOR_H
#define CELL_EXTRACTOR_H

#include "utils.h"

// ============================================================================
// EXTRACTION DES CASES DE SUDOKU
// ============================================================================

// Extrait les 81 cases (9x9) d'une grille de Sudoku normalisée
// grid: grille extraite et redressée (carré de taille égale)
// cells: tableau de sortie de 81 pointeurs d'images (chaque case normalisée à 28x28)
// Retourne true si succès
bool extract_sudoku_cells(const GrayImage *grid, GrayImage **cells);

// Nettoie une case individuelle (supprime les bordures de grille)
// cell: case brute extraite
// margin: pourcentage de marge à retirer de chaque côté (ex: 0.15 = 15%)
GrayImage* clean_cell(const GrayImage *cell, float margin);

// Détecte si une case contient un chiffre ou est vide
// Basé sur le nombre de pixels noirs après binarisation
bool is_cell_empty(const GrayImage *cell);

// Centre un chiffre dans une case (calcule le centre de masse et recentre)
GrayImage* center_digit(const GrayImage *cell);

// Normalise une case pour l'inférence CNN (28x28, [0,1])
float* prepare_cell_for_cnn(const GrayImage *cell);

#endif // CELL_EXTRACTOR_H
