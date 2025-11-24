#include "preprocessor.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// BINARISATION
// ============================================================================

void threshold_binary(GrayImage *img, uint8_t threshold) {
    size_t total = img->width * img->height;
    for (size_t i = 0; i < total; i++) {
        img->data[i] = (img->data[i] > threshold) ? 255 : 0;
    }
}

void threshold_otsu(GrayImage *img) {
    // Calcul de l'histogramme
    int histogram[256] = {0};
    size_t total = img->width * img->height;
    
    for (size_t i = 0; i < total; i++) {
        histogram[img->data[i]]++;
    }
    
    // Algorithme d'Otsu
    float sum = 0.0f;
    for (int i = 0; i < 256; i++) {
        sum += i * histogram[i];
    }
    
    float sum_b = 0.0f;
    int w_b = 0;
    int w_f = 0;
    float var_max = 0.0f;
    int threshold = 0;
    
    for (int t = 0; t < 256; t++) {
        w_b += histogram[t];
        if (w_b == 0) continue;
        
        w_f = total - w_b;
        if (w_f == 0) break;
        
        sum_b += (float)(t * histogram[t]);
        
        float m_b = sum_b / w_b;
        float m_f = (sum - sum_b) / w_f;
        
        float var_between = (float)w_b * (float)w_f * (m_b - m_f) * (m_b - m_f);
        
        if (var_between > var_max) {
            var_max = var_between;
            threshold = t;
        }
    }
    
    LOG_DEBUG("Seuil Otsu calculé: %d", threshold);
    threshold_binary(img, threshold);
}

// ============================================================================
// FILTRAGE ET DÉBRUITAGE
// ============================================================================

static float gaussian_kernel_value(int x, int y, float sigma) {
    return expf(-(x*x + y*y) / (2.0f * sigma * sigma)) / (2.0f * M_PI * sigma * sigma);
}

GrayImage* gaussian_blur(const GrayImage *img, int kernel_size, float sigma) {
    GrayImage *result = gray_image_create(img->width, img->height);
    if (!result) return NULL;
    
    int half_k = kernel_size / 2;
    
    // Créer le noyau gaussien
    float kernel[kernel_size][kernel_size];
    float sum = 0.0f;
    
    for (int i = -half_k; i <= half_k; i++) {
        for (int j = -half_k; j <= half_k; j++) {
            kernel[i + half_k][j + half_k] = gaussian_kernel_value(i, j, sigma);
            sum += kernel[i + half_k][j + half_k];
        }
    }
    
    // Normaliser le noyau
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            kernel[i][j] /= sum;
        }
    }
    
    // Appliquer la convolution
    for (size_t y = 0; y < img->height; y++) {
        for (size_t x = 0; x < img->width; x++) {
            float value = 0.0f;
            
            for (int ky = -half_k; ky <= half_k; ky++) {
                for (int kx = -half_k; kx <= half_k; kx++) {
                    int px = (int)x + kx;
                    int py = (int)y + ky;
                    
                    // Gestion des bordures (clamping)
                    px = (px < 0) ? 0 : (px >= (int)img->width ? img->width - 1 : px);
                    py = (py < 0) ? 0 : (py >= (int)img->height ? img->height - 1 : py);
                    
                    value += img->data[py * img->width + px] * kernel[ky + half_k][kx + half_k];
                }
            }
            
            result->data[y * result->width + x] = (uint8_t)clamp(value, 0, 255);
        }
    }
    
    return result;
}

static int compare_uint8(const void *a, const void *b) {
    return (*(uint8_t*)a - *(uint8_t*)b);
}

GrayImage* median_filter(const GrayImage *img, int kernel_size) {
    GrayImage *result = gray_image_create(img->width, img->height);
    if (!result) return NULL;
    
    int half_k = kernel_size / 2;
    uint8_t *window = (uint8_t*)malloc(kernel_size * kernel_size * sizeof(uint8_t));
    
    for (size_t y = 0; y < img->height; y++) {
        for (size_t x = 0; x < img->width; x++) {
            int count = 0;
            
            for (int ky = -half_k; ky <= half_k; ky++) {
                for (int kx = -half_k; kx <= half_k; kx++) {
                    int px = (int)x + kx;
                    int py = (int)y + ky;
                    
                    if (px >= 0 && px < (int)img->width && py >= 0 && py < (int)img->height) {
                        window[count++] = img->data[py * img->width + px];
                    }
                }
            }
            
            qsort(window, count, sizeof(uint8_t), compare_uint8);
            result->data[y * result->width + x] = window[count / 2];
        }
    }
    
    free(window);
    return result;
}

void dilate(GrayImage *img, int kernel_size) {
    GrayImage *temp = gray_image_clone(img);
    int half_k = kernel_size / 2;
    
    for (size_t y = 0; y < img->height; y++) {
        for (size_t x = 0; x < img->width; x++) {
            uint8_t max_val = 0;
            
            for (int ky = -half_k; ky <= half_k; ky++) {
                for (int kx = -half_k; kx <= half_k; kx++) {
                    int px = (int)x + kx;
                    int py = (int)y + ky;
                    
                    if (px >= 0 && px < (int)img->width && py >= 0 && py < (int)img->height) {
                        uint8_t val = temp->data[py * temp->width + px];
                        if (val > max_val) max_val = val;
                    }
                }
            }
            
            img->data[y * img->width + x] = max_val;
        }
    }
    
    gray_image_free(temp);
}

void erode(GrayImage *img, int kernel_size) {
    GrayImage *temp = gray_image_clone(img);
    int half_k = kernel_size / 2;
    
    for (size_t y = 0; y < img->height; y++) {
        for (size_t x = 0; x < img->width; x++) {
            uint8_t min_val = 255;
            
            for (int ky = -half_k; ky <= half_k; ky++) {
                for (int kx = -half_k; kx <= half_k; kx++) {
                    int px = (int)x + kx;
                    int py = (int)y + ky;
                    
                    if (px >= 0 && px < (int)img->width && py >= 0 && py < (int)img->height) {
                        uint8_t val = temp->data[py * temp->width + px];
                        if (val < min_val) min_val = val;
                    }
                }
            }
            
            img->data[y * img->width + x] = min_val;
        }
    }
    
    gray_image_free(temp);
}

// ============================================================================
// NORMALISATION
// ============================================================================

GrayImage* resize_image(const GrayImage *img, size_t new_width, size_t new_height) {
    GrayImage *result = gray_image_create(new_width, new_height);
    if (!result) return NULL;
    
    float x_ratio = (float)img->width / new_width;
    float y_ratio = (float)img->height / new_height;
    
    for (size_t y = 0; y < new_height; y++) {
        for (size_t x = 0; x < new_width; x++) {
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;
            
            int x0 = (int)src_x;
            int y0 = (int)src_y;
            int x1 = min_int(x0 + 1, img->width - 1);
            int y1 = min_int(y0 + 1, img->height - 1);
            
            float dx = src_x - x0;
            float dy = src_y - y0;
            
            // Interpolation bilinéaire
            float v00 = img->data[y0 * img->width + x0];
            float v10 = img->data[y0 * img->width + x1];
            float v01 = img->data[y1 * img->width + x0];
            float v11 = img->data[y1 * img->width + x1];
            
            float v0 = v00 * (1 - dx) + v10 * dx;
            float v1 = v01 * (1 - dx) + v11 * dx;
            float value = v0 * (1 - dy) + v1 * dy;
            
            result->data[y * new_width + x] = (uint8_t)clamp(value, 0, 255);
        }
    }
    
    return result;
}

float* normalize_to_float(const GrayImage *img) {
    size_t total = img->width * img->height;
    float *result = (float*)malloc(total * sizeof(float));
    
    if (!result) return NULL;
    
    for (size_t i = 0; i < total; i++) {
        result[i] = img->data[i] / 255.0f;
    }
    
    return result;
}

void invert_image(GrayImage *img) {
    size_t total = img->width * img->height;
    for (size_t i = 0; i < total; i++) {
        img->data[i] = 255 - img->data[i];
    }
}

// ============================================================================
// DÉTECTION DE CONTOURS
// ============================================================================

GrayImage* sobel_filter(const GrayImage *img) {
    GrayImage *result = gray_image_create(img->width, img->height);
    if (!result) return NULL;
    
    int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    for (size_t y = 1; y < img->height - 1; y++) {
        for (size_t x = 1; x < img->width - 1; x++) {
            float gx = 0.0f, gy = 0.0f;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    uint8_t val = img->data[py * img->width + px];
                    
                    gx += val * sobel_x[ky + 1][kx + 1];
                    gy += val * sobel_y[ky + 1][kx + 1];
                }
            }
            
            float magnitude = sqrtf(gx * gx + gy * gy);
            result->data[y * result->width + x] = (uint8_t)clamp(magnitude, 0, 255);
        }
    }
    
    return result;
}

GrayImage* canny_edge_detection(const GrayImage *img, float low_threshold, float high_threshold) {
    // Simplification: appliquer Sobel puis double seuillage
    GrayImage *edges = sobel_filter(img);
    if (!edges) return NULL;
    
    // Double seuillage
    for (size_t i = 0; i < edges->width * edges->height; i++) {
        uint8_t val = edges->data[i];
        if (val > high_threshold) {
            edges->data[i] = 255;
        } else if (val > low_threshold) {
            edges->data[i] = 128;  // Candidat faible
        } else {
            edges->data[i] = 0;
        }
    }
    
    // Hystérésis simplifiée (connecter les arêtes faibles aux arêtes fortes)
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t y = 1; y < edges->height - 1; y++) {
            for (size_t x = 1; x < edges->width - 1; x++) {
                if (edges->data[y * edges->width + x] == 128) {
                    bool has_strong_neighbor = false;
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (edges->data[(y + dy) * edges->width + (x + dx)] == 255) {
                                has_strong_neighbor = true;
                                break;
                            }
                        }
                    }
                    if (has_strong_neighbor) {
                        edges->data[y * edges->width + x] = 255;
                        changed = true;
                    }
                }
            }
        }
    }
    
    // Nettoyer les arêtes faibles restantes
    for (size_t i = 0; i < edges->width * edges->height; i++) {
        if (edges->data[i] == 128) edges->data[i] = 0;
    }
    
    return edges;
}
