#include "cell_extractor.h"
#include "preprocessor.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static GrayImage* crop_image(const GrayImage *src, int x, int y, int w, int h) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x + w > (int)src->width) w = src->width - x;
    if (y + h > (int)src->height) h = src->height - y;
    
    if (w <= 0 || h <= 0) return NULL;
    
    GrayImage *dst = (GrayImage*)malloc(sizeof(GrayImage));
    dst->width = w;
    dst->height = h;
    dst->data = (uint8_t*)malloc(w * h * sizeof(uint8_t));
    
    for (int i = 0; i < h; i++) {
        memcpy(dst->data + i * w, src->data + (y + i) * src->width + x, w);
    }
    
    return dst;
}

static void get_center_of_mass(const GrayImage *img, float *cx, float *cy) {
    float sum_x = 0, sum_y = 0, sum_w = 0;
    
    for (size_t y = 0; y < img->height; y++) {
        for (size_t x = 0; x < img->width; x++) {
            // Assuming white digit on black background or inverted?
            // Usually MNIST is white on black.
            // If input is black on white (paper), we should invert or check pixel value.
            // Let's assume pixel value represents "mass" (brightness).
            // If digit is dark (0) and background white (255), we should invert.
            // But usually we work with inverted images (digit white).
            
            uint8_t val = img->data[y * img->width + x];
            sum_x += x * val;
            sum_y += y * val;
            sum_w += val;
        }
    }
    
    if (sum_w > 0) {
        *cx = sum_x / sum_w;
        *cy = sum_y / sum_w;
    } else {
        *cx = img->width / 2.0f;
        *cy = img->height / 2.0f;
    }
}

// ============================================================================
// CELL EXTRACTION
// ============================================================================

GrayImage* clean_cell(const GrayImage *cell, float margin) {
    int w = cell->width;
    int h = cell->height;
    
    int margin_x = (int)(w * margin);
    int margin_y = (int)(h * margin);
    
    // Crop inner part
    return crop_image(cell, margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y);
}

bool extract_sudoku_cells(const GrayImage *grid, GrayImage **cells) {
    if (!grid || !cells) return false;
    
    int cell_w = grid->width / 9;
    int cell_h = grid->height / 9;
    
    // Increased margin to 20% to avoid grid lines as requested
    float margin = 0.20f; 
    
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            int x = col * cell_w;
            int y = row * cell_h;
            
            // Extract raw cell
            GrayImage *raw = crop_image(grid, x, y, cell_w, cell_h);
            if (!raw) return false;
            
            // Clean (crop margin)
            GrayImage *cleaned = clean_cell(raw, margin);
            free(raw->data); free(raw);
            
            // Resize to 28x28 for CNN
            GrayImage *resized = resize_image(cleaned, 28, 28);
            free(cleaned->data); free(cleaned);
            
            // Center digit (optional but good for CNN)
            GrayImage *centered = center_digit(resized);
            free(resized->data); free(resized);
            
            cells[row * 9 + col] = centered;
        }
    }
    
    return true;
}

bool is_cell_empty(const GrayImage *cell) {
    // Count non-zero pixels
    int count = 0;
    int total = cell->width * cell->height;
    
    for (int i = 0; i < total; i++) {
        if (cell->data[i] > 128) count++; // Assuming white digit
    }
    
    // If less than 5% pixels are white, it's empty
    return (float)count / total < 0.05f;
}

GrayImage* center_digit(const GrayImage *cell) {
    float cx, cy;
    get_center_of_mass(cell, &cx, &cy);
    
    int dx = (int)(cell->width / 2.0f - cx);
    int dy = (int)(cell->height / 2.0f - cy);
    
    GrayImage *dst = (GrayImage*)malloc(sizeof(GrayImage));
    dst->width = cell->width;
    dst->height = cell->height;
    dst->data = (uint8_t*)calloc(dst->width * dst->height, sizeof(uint8_t)); // Init to black
    
    for (size_t y = 0; y < cell->height; y++) {
        for (size_t x = 0; x < cell->width; x++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < (int)dst->width && ny >= 0 && ny < (int)dst->height) {
                dst->data[ny * dst->width + nx] = cell->data[y * cell->width + x];
            }
        }
    }
    
    return dst;
}

float* prepare_cell_for_cnn(const GrayImage *cell) {
    return normalize_to_float(cell);
}
