#ifndef PERSPECTIVE_H
#define PERSPECTIVE_H

#include "utils.h"
#include "grid_detector.h"

// ============================================================================
// TRANSFORMATION PERSPECTIVE
// ============================================================================

// Matrice 3x3 pour transformation homographique
typedef struct {
    float data[3][3];
} HomographyMatrix;

// Calcule la matrice d'homographie pour mapper un quadrilatère source vers un rectangle dest
// src: quadrilatère source (4 coins)
// dst: rectangle destination (4 coins)
HomographyMatrix compute_homography(const Quad *src, const Quad *dst);

// Applique une transformation perspective à une image
// img: image source
// H: matrice d'homographie
// output_width, output_height: dimensions de l'image de sortie
GrayImage* warp_perspective(const GrayImage *img, const HomographyMatrix *H, 
                            size_t output_width, size_t output_height);

// Transforme un point avec une matrice d'homographie
Point2D transform_point(const HomographyMatrix *H, Point2D point);

// Inverse une matrice 3x3 (nécessaire pour certaines transformations)
bool invert_matrix_3x3(const float src[3][3], float dst[3][3]);

// ============================================================================
// UTILITAIRES
// ============================================================================

// Crée un quadrilatère rectangle (pour destination de warp)
Quad make_rectangle_quad(float width, float height);

// Extrait une grille de Sudoku depuis l'image originale
// quad: coins détectés de la grille
// output_size: taille de sortie (ex: 450x450 pixels)
GrayImage* extract_grid(const GrayImage *img, const Quad *quad, size_t output_size);

#endif // PERSPECTIVE_H
