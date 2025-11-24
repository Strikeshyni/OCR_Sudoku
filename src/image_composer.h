#ifndef IMAGE_COMPOSER_H
#define IMAGE_COMPOSER_H

#include "utils.h"
#include "sudoku_solver.h"
#include "grid_detector.h"

// ============================================================================
// COMPOSITION D'IMAGE
// ============================================================================

// Dessine un chiffre sur une image à une position donnée
// img: image destination
// digit: chiffre à dessiner (1-9)
// x, y: position top-left du chiffre
// size: taille du chiffre en pixels
// color: couleur (0-255)
void draw_digit(GrayImage *img, int digit, int x, int y, int size, uint8_t color);

// Génère une image des chiffres manquants incrustés sur la grille originale
// original: image originale (ou grille extraite)
// original_grid: grille originale (avec cases vides)
// solved_grid: grille résolue
// quad: quadrilatère de la grille détectée (pour transformation inverse)
RGBImage* compose_solved_image(const GrayImage *original, 
                               const SudokuGrid *original_grid,
                               const SudokuGrid *solved_grid,
                               const Quad *quad);

// Version simplifiée: compose sur une grille normalisée (sans transformation inverse)
RGBImage* compose_solved_grid(const GrayImage *grid_image,
                              const SudokuGrid *original_grid,
                              const SudokuGrid *solved_grid);

// ============================================================================
// DESSIN DE PRIMITIVES
// ============================================================================

// Dessine une ligne sur une image RGB
void draw_line_rgb(RGBImage *img, Point2D p1, Point2D p2, 
                   uint8_t r, uint8_t g, uint8_t b, int thickness);

// Dessine un rectangle
void draw_rectangle(RGBImage *img, int x, int y, int width, int height,
                   uint8_t r, uint8_t g, uint8_t b, int thickness);

// Remplit un rectangle
void fill_rectangle(RGBImage *img, int x, int y, int width, int height,
                   uint8_t r, uint8_t g, uint8_t b);

// ============================================================================
// FONTE (CHIFFRES BITMAP 7-SEGMENTS SIMPLIFIÉ)
// ============================================================================

// Dessine un chiffre avec une fonte bitmap simple
void draw_digit_bitmap(GrayImage *img, int digit, int x, int y, int size);

// Dessine un chiffre sur une image RGB
void draw_digit_rgb(RGBImage *img, int digit, int x, int y, int size,
                   uint8_t r, uint8_t g, uint8_t b);

#endif // IMAGE_COMPOSER_H
