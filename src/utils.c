#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

// ============================================================================
// GESTION MÉMOIRE MATRICES
// ============================================================================

Matrix* matrix_create(size_t rows, size_t cols) {
    Matrix *mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) return NULL;
    
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (float*)calloc(rows * cols, sizeof(float));
    
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    
    return mat;
}

void matrix_free(Matrix *mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

void matrix_fill(Matrix *mat, float value) {
    size_t total = mat->rows * mat->cols;
    for (size_t i = 0; i < total; i++) {
        mat->data[i] = value;
    }
}

void matrix_randomize(Matrix *mat, float min, float max) {
    static bool seeded = false;
    if (!seeded) {
        srand(time(NULL));
        seeded = true;
    }
    
    size_t total = mat->rows * mat->cols;
    for (size_t i = 0; i < total; i++) {
        mat->data[i] = randf(min, max);
    }
}

void matrix_copy(const Matrix *src, Matrix *dst) {
    if (src->rows != dst->rows || src->cols != dst->cols) {
        LOG_ERROR("matrix_copy: dimensions mismatch");
        return;
    }
    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(float));
}

Matrix* matrix_clone(const Matrix *src) {
    Matrix *clone = matrix_create(src->rows, src->cols);
    if (clone) {
        matrix_copy(src, clone);
    }
    return clone;
}

// ============================================================================
// OPÉRATIONS MATRICIELLES
// ============================================================================

void matrix_add(const Matrix *a, const Matrix *b, Matrix *result) {
    size_t total = a->rows * a->cols;
    for (size_t i = 0; i < total; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
}

void matrix_subtract(const Matrix *a, const Matrix *b, Matrix *result) {
    size_t total = a->rows * a->cols;
    for (size_t i = 0; i < total; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
}

void matrix_multiply(const Matrix *a, const Matrix *b, Matrix *result) {
    // Standard matrix multiplication: C = A * B
    // A: (m x n), B: (n x p) -> C: (m x p)
    if (a->cols != b->rows) {
        LOG_ERROR("matrix_multiply: incompatible dimensions");
        return;
    }
    
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result->data[i * result->cols + j] = sum;
        }
    }
}

void matrix_elementwise_multiply(const Matrix *a, const Matrix *b, Matrix *result) {
    size_t total = a->rows * a->cols;
    for (size_t i = 0; i < total; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
}

void matrix_scale(Matrix *mat, float scalar) {
    size_t total = mat->rows * mat->cols;
    for (size_t i = 0; i < total; i++) {
        mat->data[i] *= scalar;
    }
}

void matrix_transpose(const Matrix *src, Matrix *dst) {
    if (src->rows != dst->cols || src->cols != dst->rows) {
        LOG_ERROR("matrix_transpose: incompatible dimensions");
        return;
    }
    
    for (size_t i = 0; i < src->rows; i++) {
        for (size_t j = 0; j < src->cols; j++) {
            dst->data[j * dst->cols + i] = src->data[i * src->cols + j];
        }
    }
}

// ============================================================================
// GESTION MÉMOIRE IMAGES
// ============================================================================

GrayImage* gray_image_create(size_t width, size_t height) {
    GrayImage *img = (GrayImage*)malloc(sizeof(GrayImage));
    if (!img) return NULL;
    
    img->width = width;
    img->height = height;
    img->data = (uint8_t*)calloc(width * height, sizeof(uint8_t));
    
    if (!img->data) {
        free(img);
        return NULL;
    }
    
    return img;
}

void gray_image_free(GrayImage *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

GrayImage* gray_image_clone(const GrayImage *src) {
    GrayImage *clone = gray_image_create(src->width, src->height);
    if (clone) {
        memcpy(clone->data, src->data, src->width * src->height);
    }
    return clone;
}

RGBImage* rgb_image_create(size_t width, size_t height, size_t channels) {
    RGBImage *img = (RGBImage*)malloc(sizeof(RGBImage));
    if (!img) return NULL;
    
    img->width = width;
    img->height = height;
    img->channels = channels;
    img->data = (uint8_t*)calloc(width * height * channels, sizeof(uint8_t));
    
    if (!img->data) {
        free(img);
        return NULL;
    }
    
    return img;
}

void rgb_image_free(RGBImage *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

// ============================================================================
// FONCTIONS MATHÉMATIQUES
// ============================================================================

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

float tanh_activation(float x) {
    return tanhf(x);
}

float tanh_derivative(float x) {
    float t = tanhf(x);
    return 1.0f - t * t;
}

void softmax(const float *input, float *output, size_t length) {
    float max_val = input[0];
    for (size_t i = 1; i < length; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < length; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    for (size_t i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

float cross_entropy_loss(const float *predicted, const float *target, size_t length) {
    float loss = 0.0f;
    for (size_t i = 0; i < length; i++) {
        loss -= target[i] * logf(predicted[i] + 1e-7f);
    }
    return loss;
}

// ============================================================================
// UTILITAIRES
// ============================================================================

float randf(float min, float max) {
    return min + (max - min) * ((float)rand() / (float)RAND_MAX);
}

int rand_int(int min, int max) {
    return min + (rand() % (max - min + 1));
}

void shuffle_indices(int *indices, size_t count) {
    for (size_t i = count - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

int min_int(int a, int b) {
    return a < b ? a : b;
}

int max_int(int a, int b) {
    return a > b ? a : b;
}

float min_float(float a, float b) {
    return a < b ? a : b;
}

float max_float(float a, float b) {
    return a > b ? a : b;
}

// ============================================================================
// DEBUG ET LOGGING
// ============================================================================

void print_matrix(const Matrix *mat, const char *name) {
    printf("Matrix %s (%zux%zu):\n", name, mat->rows, mat->cols);
    for (size_t i = 0; i < mat->rows && i < 5; i++) {
        for (size_t j = 0; j < mat->cols && j < 10; j++) {
            printf("%.4f ", mat->data[i * mat->cols + j]);
        }
        if (mat->cols > 10) printf("...");
        printf("\n");
    }
    if (mat->rows > 5) printf("...\n");
}

void print_image_stats(const GrayImage *img, const char *name) {
    uint8_t min_val = 255, max_val = 0;
    uint32_t sum = 0;
    
    size_t total = img->width * img->height;
    for (size_t i = 0; i < total; i++) {
        uint8_t val = img->data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }
    
    printf("Image %s (%zux%zu): min=%u, max=%u, mean=%.2f\n",
           name, img->width, img->height, min_val, max_val, (float)sum / total);
}
