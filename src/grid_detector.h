#ifndef GRID_DETECTOR_H
#define GRID_DETECTOR_H

#include "utils.h"

// ============================================================================
// STRUCTURES
// ============================================================================

// Ligne en coordonnées polaires (rho, theta) pour transformée de Hough
typedef struct {
    float rho;
    float theta;
    int votes;
} HoughLine;

// Quadrilatère (4 points)
typedef struct {
    Point2D corners[4];
} Quad;

// ============================================================================
// DÉTECTION DE LIGNES (HOUGH TRANSFORM)
// ============================================================================

// Transformée de Hough pour détecter les lignes
// Retourne un tableau de lignes détectées et leur nombre dans *num_lines
HoughLine* hough_lines(const GrayImage *edges, int threshold, int *num_lines);

// ============================================================================
// DÉTECTION DE CONTOURS
// ============================================================================

// Trouve les contours dans une image binaire
// Retourne le nombre de composantes connexes
int find_contours(const GrayImage *binary, Point2D **contours, int **contour_sizes);

// ============================================================================
// DÉTECTION DE QUADRILATÈRE
// ============================================================================

// Trouve le plus grand quadrilatère dans l'image (grille de Sudoku)
// Utilise les contours et approximation polygonale
bool find_largest_quad(const GrayImage *edges, Quad *quad);

// Méthode alternative: détection par projection horizontale/verticale
bool find_grid_by_projection(const GrayImage *edges, Quad *quad);

// Ordonne les coins du quadrilatère (top-left, top-right, bottom-right, bottom-left)
void order_quad_corners(Quad *quad);

// ============================================================================
// UTILITAIRES
// ============================================================================

// Calcule l'aire d'un quadrilatère
float quad_area(const Quad *quad);

// Vérifie si un quadrilatère est valide (angles raisonnables, aire suffisante)
bool is_valid_quad(const Quad *quad, size_t img_width, size_t img_height);

#endif // GRID_DETECTOR_H
