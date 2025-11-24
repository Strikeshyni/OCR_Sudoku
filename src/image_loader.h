#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include "utils.h"

// ============================================================================
// CHARGEMENT ET SAUVEGARDE D'IMAGES
// ============================================================================

// Charge une image RGB depuis un fichier (JPG, PNG, BMP, etc.)
// Retourne NULL en cas d'échec
RGBImage* load_rgb_image(const char *filename);

// Charge une image directement en niveaux de gris
GrayImage* load_gray_image(const char *filename);

// Sauvegarde une image RGB en PNG
bool save_rgb_image(const char *filename, const RGBImage *img);

// Sauvegarde une image en niveaux de gris en PNG
bool save_gray_image(const char *filename, const GrayImage *img);

// ============================================================================
// CONVERSIONS
// ============================================================================

// Convertit RGB vers niveaux de gris (formule standard: 0.299*R + 0.587*G + 0.114*B)
GrayImage* rgb_to_gray(const RGBImage *rgb);

// Convertit niveaux de gris vers RGB (copie sur les 3 canaux)
RGBImage* gray_to_rgb(const GrayImage *gray);

#endif // IMAGE_LOADER_H
