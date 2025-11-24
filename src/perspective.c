#include "perspective.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// ============================================================================
// CALCUL D'HOMOGRAPHIE
// ============================================================================

// Résout un système linéaire Ax = b par élimination de Gauss
static bool solve_linear_system(float A[8][8], float b[8], float x[8]) {
    // Élimination avant
    for (int i = 0; i < 8; i++) {
        // Trouver le pivot
        int max_row = i;
        for (int k = i + 1; k < 8; k++) {
            if (fabsf(A[k][i]) > fabsf(A[max_row][i])) {
                max_row = k;
            }
        }
        
        // Échanger les lignes
        if (max_row != i) {
            for (int k = 0; k < 8; k++) {
                float temp = A[i][k];
                A[i][k] = A[max_row][k];
                A[max_row][k] = temp;
            }
            float temp = b[i];
            b[i] = b[max_row];
            b[max_row] = temp;
        }
        
        if (fabsf(A[i][i]) < 1e-10f) {
            return false;  // Matrice singulière
        }
        
        // Élimination
        for (int k = i + 1; k < 8; k++) {
            float factor = A[k][i] / A[i][i];
            for (int j = i; j < 8; j++) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }
    
    // Substitution arrière
    for (int i = 7; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < 8; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
    
    return true;
}

HomographyMatrix compute_homography(const Quad *src, const Quad *dst) {
    // Résoudre le système pour trouver h00, h01, ..., h21 (h22 = 1)
    // Chaque correspondance de point donne 2 équations
    
    float A[8][8] = {0};
    float b[8] = {0};
    
    for (int i = 0; i < 4; i++) {
        float x = src->corners[i].x;
        float y = src->corners[i].y;
        float u = dst->corners[i].x;
        float v = dst->corners[i].y;
        
        int row = i * 2;
        
        // Première équation (pour u)
        A[row][0] = x;
        A[row][1] = y;
        A[row][2] = 1;
        A[row][3] = 0;
        A[row][4] = 0;
        A[row][5] = 0;
        A[row][6] = -u * x;
        A[row][7] = -u * y;
        b[row] = u;
        
        // Deuxième équation (pour v)
        A[row + 1][0] = 0;
        A[row + 1][1] = 0;
        A[row + 1][2] = 0;
        A[row + 1][3] = x;
        A[row + 1][4] = y;
        A[row + 1][5] = 1;
        A[row + 1][6] = -v * x;
        A[row + 1][7] = -v * y;
        b[row + 1] = v;
    }
    
    float h[8];
    solve_linear_system(A, b, h);
    
    HomographyMatrix H;
    H.data[0][0] = h[0]; H.data[0][1] = h[1]; H.data[0][2] = h[2];
    H.data[1][0] = h[3]; H.data[1][1] = h[4]; H.data[1][2] = h[5];
    H.data[2][0] = h[6]; H.data[2][1] = h[7]; H.data[2][2] = 1.0f;
    
    return H;
}

// ============================================================================
// TRANSFORMATION D'IMAGE
// ============================================================================

Point2D transform_point(const HomographyMatrix *H, Point2D point) {
    float x = point.x;
    float y = point.y;
    
    float w = H->data[2][0] * x + H->data[2][1] * y + H->data[2][2];
    
    Point2D result;
    result.x = (H->data[0][0] * x + H->data[0][1] * y + H->data[0][2]) / w;
    result.y = (H->data[1][0] * x + H->data[1][1] * y + H->data[1][2]) / w;
    
    return result;
}

GrayImage* warp_perspective(const GrayImage *img, const HomographyMatrix *H, 
                            size_t output_width, size_t output_height) {
    GrayImage *result = gray_image_create(output_width, output_height);
    if (!result) return NULL;
    
    // Calculer l'inverse de H pour faire le mapping inverse
    float H_inv_data[3][3];
    if (!invert_matrix_3x3(H->data, H_inv_data)) {
        LOG_ERROR("Impossible d'inverser la matrice d'homographie");
        gray_image_free(result);
        return NULL;
    }
    
    HomographyMatrix H_inv;
    memcpy(H_inv.data, H_inv_data, sizeof(H_inv_data));
    
    // Pour chaque pixel de l'image de sortie, trouver le pixel correspondant dans l'image source
    for (size_t y = 0; y < output_height; y++) {
        for (size_t x = 0; x < output_width; x++) {
            Point2D dst_point = {x, y};
            Point2D src_point = transform_point(&H_inv, dst_point);
            
            int sx = (int)roundf(src_point.x);
            int sy = (int)roundf(src_point.y);
            
            if (sx >= 0 && sx < (int)img->width && sy >= 0 && sy < (int)img->height) {
                result->data[y * output_width + x] = img->data[sy * img->width + sx];
            } else {
                result->data[y * output_width + x] = 0;  // Fond noir
            }
        }
    }
    
    return result;
}

// ============================================================================
// INVERSION DE MATRICE 3x3
// ============================================================================

bool invert_matrix_3x3(const float src[3][3], float dst[3][3]) {
    float det = src[0][0] * (src[1][1] * src[2][2] - src[1][2] * src[2][1])
              - src[0][1] * (src[1][0] * src[2][2] - src[1][2] * src[2][0])
              + src[0][2] * (src[1][0] * src[2][1] - src[1][1] * src[2][0]);
    
    if (fabsf(det) < 1e-10f) {
        return false;  // Matrice non inversible
    }
    
    float inv_det = 1.0f / det;
    
    dst[0][0] = (src[1][1] * src[2][2] - src[1][2] * src[2][1]) * inv_det;
    dst[0][1] = (src[0][2] * src[2][1] - src[0][1] * src[2][2]) * inv_det;
    dst[0][2] = (src[0][1] * src[1][2] - src[0][2] * src[1][1]) * inv_det;
    
    dst[1][0] = (src[1][2] * src[2][0] - src[1][0] * src[2][2]) * inv_det;
    dst[1][1] = (src[0][0] * src[2][2] - src[0][2] * src[2][0]) * inv_det;
    dst[1][2] = (src[0][2] * src[1][0] - src[0][0] * src[1][2]) * inv_det;
    
    dst[2][0] = (src[1][0] * src[2][1] - src[1][1] * src[2][0]) * inv_det;
    dst[2][1] = (src[0][1] * src[2][0] - src[0][0] * src[2][1]) * inv_det;
    dst[2][2] = (src[0][0] * src[1][1] - src[0][1] * src[1][0]) * inv_det;
    
    return true;
}

// ============================================================================
// UTILITAIRES
// ============================================================================

Quad make_rectangle_quad(float width, float height) {
    Quad quad;
    quad.corners[0] = (Point2D){0, 0};              // Top-left
    quad.corners[1] = (Point2D){width, 0};          // Top-right
    quad.corners[2] = (Point2D){width, height};     // Bottom-right
    quad.corners[3] = (Point2D){0, height};         // Bottom-left
    return quad;
}

GrayImage* extract_grid(const GrayImage *img, const Quad *quad, size_t output_size) {
    Quad dst_quad = make_rectangle_quad(output_size, output_size);
    HomographyMatrix H = compute_homography(quad, &dst_quad);
    
    GrayImage *warped = warp_perspective(img, &H, output_size, output_size);
    
    if (warped) {
        LOG_INFO("Grille extraite et redressée (%zux%zu)", output_size, output_size);
    }
    
    return warped;
}
