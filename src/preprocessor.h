#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include "utils.h"

// ============================================================================
// BINARISATION
// ============================================================================

// Binarisation simple avec seuil fixe
void threshold_binary(GrayImage *img, uint8_t threshold);

// Binarisation adaptative (Otsu)
// Calcule automatiquement le seuil optimal
void threshold_otsu(GrayImage *img);

// ============================================================================
// FILTRAGE ET DÉBRUITAGE
// ============================================================================

// Filtre gaussien pour réduire le bruit
GrayImage* gaussian_blur(const GrayImage *img, int kernel_size, float sigma);

// Filtre médian (très efficace contre le bruit sel-poivre)
GrayImage* median_filter(const GrayImage *img, int kernel_size);

// Dilatation morphologique (épaissit les objets blancs)
void dilate(GrayImage *img, int kernel_size);

// Érosion morphologique (amincit les objets blancs)
void erode(GrayImage *img, int kernel_size);

// ============================================================================
// NORMALISATION
// ============================================================================

// Redimensionne une image (interpolation bilinéaire)
GrayImage* resize_image(const GrayImage *img, size_t new_width, size_t new_height);

// Normalise les pixels dans [0, 1] et retourne un tableau de floats
float* normalize_to_float(const GrayImage *img);

// Inverse les couleurs (noir <-> blanc)
void invert_image(GrayImage *img);

// ============================================================================
// DÉTECTION DE CONTOURS
// ============================================================================

// Filtre de Sobel pour détecter les contours (retourne gradients)
GrayImage* sobel_filter(const GrayImage *img);

// Détection de contours Canny complet
GrayImage* canny_edge_detection(const GrayImage *img, float low_threshold, float high_threshold);

#endif // PREPROCESSOR_H
