#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// ============================================================================
// STRUCTURES DE DONNÉES
// ============================================================================

// Matrice générique
typedef struct {
    float *data;
    size_t rows;
    size_t cols;
} Matrix;

// Image en niveaux de gris
typedef struct {
    uint8_t *data;
    size_t width;
    size_t height;
} GrayImage;

// Image RGB
typedef struct {
    uint8_t *data;  // RGB entrelacé
    size_t width;
    size_t height;
    size_t channels;  // 3 pour RGB, 4 pour RGBA
} RGBImage;

// Point 2D
typedef struct {
    float x;
    float y;
} Point2D;

// ============================================================================
// GESTION MÉMOIRE MATRICES
// ============================================================================

Matrix* matrix_create(size_t rows, size_t cols);
void matrix_free(Matrix *mat);
void matrix_fill(Matrix *mat, float value);
void matrix_randomize(Matrix *mat, float min, float max);
void matrix_copy(const Matrix *src, Matrix *dst);
Matrix* matrix_clone(const Matrix *src);

// ============================================================================
// OPÉRATIONS MATRICIELLES
// ============================================================================

void matrix_add(const Matrix *a, const Matrix *b, Matrix *result);
void matrix_subtract(const Matrix *a, const Matrix *b, Matrix *result);
void matrix_multiply(const Matrix *a, const Matrix *b, Matrix *result);
void matrix_elementwise_multiply(const Matrix *a, const Matrix *b, Matrix *result);
void matrix_scale(Matrix *mat, float scalar);
void matrix_transpose(const Matrix *src, Matrix *dst);

// ============================================================================
// GESTION MÉMOIRE IMAGES
// ============================================================================

GrayImage* gray_image_create(size_t width, size_t height);
void gray_image_free(GrayImage *img);
GrayImage* gray_image_clone(const GrayImage *src);

RGBImage* rgb_image_create(size_t width, size_t height, size_t channels);
void rgb_image_free(RGBImage *img);

// ============================================================================
// FONCTIONS MATHÉMATIQUES
// ============================================================================

float relu(float x);
float relu_derivative(float x);
float sigmoid(float x);
float sigmoid_derivative(float x);
float tanh_activation(float x);
float tanh_derivative(float x);

void softmax(const float *input, float *output, size_t length);
float cross_entropy_loss(const float *predicted, const float *target, size_t length);

// ============================================================================
// UTILITAIRES
// ============================================================================

float randf(float min, float max);
int rand_int(int min, int max);
void shuffle_indices(int *indices, size_t count);

float clamp(float value, float min, float max);
int min_int(int a, int b);
int max_int(int a, int b);
float min_float(float a, float b);
float max_float(float a, float b);

// ============================================================================
// DEBUG ET LOGGING
// ============================================================================

void print_matrix(const Matrix *mat, const char *name);
void print_image_stats(const GrayImage *img, const char *name);

#ifdef DEBUG
    #define LOG_DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
    #define LOG_DEBUG(fmt, ...) 
#endif

#define LOG_INFO(fmt, ...) fprintf(stdout, "[INFO] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

#endif // UTILS_H
