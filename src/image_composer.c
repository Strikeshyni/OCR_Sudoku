#include "image_composer.h"
#include "image_loader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// FONTE 7-SEGMENTS (CHIFFRES 0-9)
// ============================================================================

// Représentation bitmap 7-segments simplifié (5x7 pixels par chiffre)
static const uint8_t digit_bitmaps[10][7] = {
    // 0
    {0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110},
    // 1
    {0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110},
    // 2
    {0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111},
    // 3
    {0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110},
    // 4
    {0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010},
    // 5
    {0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110},
    // 6
    {0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110},
    // 7
    {0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000},
    // 8
    {0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110},
    // 9
    {0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110}
};

void draw_digit_bitmap(GrayImage *img, int digit, int x, int y, int size) {
    if (digit < 0 || digit > 9) return;
    
    const uint8_t *bitmap = digit_bitmaps[digit];
    int scale = size / 7;  // Facteur d'échelle
    
    for (int row = 0; row < 7; row++) {
        for (int col = 0; col < 5; col++) {
            if (bitmap[row] & (1 << (4 - col))) {
                // Dessiner un bloc scalé
                for (int dy = 0; dy < scale; dy++) {
                    for (int dx = 0; dx < scale; dx++) {
                        int px = x + col * scale + dx;
                        int py = y + row * scale + dy;
                        
                        if (px >= 0 && px < (int)img->width && 
                            py >= 0 && py < (int)img->height) {
                            img->data[py * img->width + px] = 255;
                        }
                    }
                }
            }
        }
    }
}

void draw_digit_rgb(RGBImage *img, int digit, int x, int y, int size,
                   uint8_t r, uint8_t g, uint8_t b) {
    if (digit < 0 || digit > 9) return;
    
    const uint8_t *bitmap = digit_bitmaps[digit];
    int scale = size / 7;
    
    for (int row = 0; row < 7; row++) {
        for (int col = 0; col < 5; col++) {
            if (bitmap[row] & (1 << (4 - col))) {
                for (int dy = 0; dy < scale; dy++) {
                    for (int dx = 0; dx < scale; dx++) {
                        int px = x + col * scale + dx;
                        int py = y + row * scale + dy;
                        
                        if (px >= 0 && px < (int)img->width && 
                            py >= 0 && py < (int)img->height) {
                            int idx = (py * img->width + px) * img->channels;
                            img->data[idx] = r;
                            img->data[idx + 1] = g;
                            img->data[idx + 2] = b;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// COMPOSITION DE GRILLE RÉSOLUE
// ============================================================================

RGBImage* compose_solved_grid(const GrayImage *grid_image,
                              const SudokuGrid *original_grid,
                              const SudokuGrid *solved_grid) {
    
    // Convertir l'image de grille en RGB
    RGBImage *result = gray_to_rgb(grid_image);
    if (!result) return NULL;
    
    int cell_size = grid_image->width / 9;
    int digit_size = cell_size * 2 / 3;  // 2/3 de la taille de la case
    
    // Pour chaque case
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            // Si la case était vide dans la grille originale
            if (!original_grid->fixed[row][col]) {
                int digit = solved_grid->grid[row][col];
                
                if (digit >= 1 && digit <= 9) {
                    // Calculer la position centrée du chiffre
                    int x = col * cell_size + (cell_size - digit_size * 5 / 7) / 2;
                    int y = row * cell_size + (cell_size - digit_size) / 2;
                    
                    // Dessiner le chiffre en rouge (pour différencier des originaux)
                    draw_digit_rgb(result, digit, x, y, digit_size, 255, 0, 0);
                }
            }
        }
    }
    
    LOG_INFO("Image de grille résolue composée");
    return result;
}

RGBImage* compose_solved_image(const GrayImage *original, 
                               const SudokuGrid *original_grid,
                               const SudokuGrid *solved_grid,
                               const Quad *quad) {
    
    // Pour simplifier, on utilise la version sans transformation inverse
    // Une implémentation complète appliquerait la transformation perspective inverse
    // pour replacer les chiffres exactement sur l'image originale
    
    LOG_INFO("Composition de l'image finale...");
    
    // Version simplifiée: juste composer sur la grille normalisée
    return compose_solved_grid(original, original_grid, solved_grid);
}

// ============================================================================
// DESSIN DE PRIMITIVES
// ============================================================================

void draw_line_rgb(RGBImage *img, Point2D p1, Point2D p2, 
                   uint8_t r, uint8_t g, uint8_t b, int thickness) {
    
    // Algorithme de Bresenham
    int x0 = (int)p1.x, y0 = (int)p1.y;
    int x1 = (int)p2.x, y1 = (int)p2.y;
    
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    
    while (true) {
        // Dessiner un point avec épaisseur
        for (int ty = -thickness/2; ty <= thickness/2; ty++) {
            for (int tx = -thickness/2; tx <= thickness/2; tx++) {
                int px = x0 + tx;
                int py = y0 + ty;
                
                if (px >= 0 && px < (int)img->width && 
                    py >= 0 && py < (int)img->height) {
                    int idx = (py * img->width + px) * img->channels;
                    img->data[idx] = r;
                    img->data[idx + 1] = g;
                    img->data[idx + 2] = b;
                }
            }
        }
        
        if (x0 == x1 && y0 == y1) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

void draw_rectangle(RGBImage *img, int x, int y, int width, int height,
                   uint8_t r, uint8_t g, uint8_t b, int thickness) {
    Point2D tl = {x, y};
    Point2D tr = {x + width, y};
    Point2D br = {x + width, y + height};
    Point2D bl = {x, y + height};
    
    draw_line_rgb(img, tl, tr, r, g, b, thickness);
    draw_line_rgb(img, tr, br, r, g, b, thickness);
    draw_line_rgb(img, br, bl, r, g, b, thickness);
    draw_line_rgb(img, bl, tl, r, g, b, thickness);
}

void fill_rectangle(RGBImage *img, int x, int y, int width, int height,
                   uint8_t r, uint8_t g, uint8_t b) {
    for (int py = y; py < y + height; py++) {
        for (int px = x; px < x + width; px++) {
            if (px >= 0 && px < (int)img->width && 
                py >= 0 && py < (int)img->height) {
                int idx = (py * img->width + px) * img->channels;
                img->data[idx] = r;
                img->data[idx + 1] = g;
                img->data[idx + 2] = b;
            }
        }
    }
}
