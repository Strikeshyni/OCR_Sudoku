#include "dataset_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// UTILITAIRES IDX
// ============================================================================

static uint32_t read_big_endian_int(FILE *file) {
    uint8_t bytes[4];
    size_t read = fread(bytes, 1, 4, file);
    if (read != 4) {
        LOG_ERROR("Erreur de lecture (lu %zu octets au lieu de 4)", read);
        return 0;
    }
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// ============================================================================
// CHARGEMENT MNIST
// ============================================================================

MNISTDataset* load_mnist_dataset(const char *images_path, const char *labels_path) {
    // Ouvrir le fichier d'images
    FILE *images_file = fopen(images_path, "rb");
    if (!images_file) {
        LOG_ERROR("Impossible d'ouvrir: %s", images_path);
        return NULL;
    }
    
    // Lire l'en-tête des images
    uint32_t magic_images = read_big_endian_int(images_file);
    if (magic_images != 2051) {
        LOG_ERROR("Format IDX invalide pour les images");
        fclose(images_file);
        return NULL;
    }
    
    uint32_t num_images = read_big_endian_int(images_file);
    uint32_t rows = read_big_endian_int(images_file);
    uint32_t cols = read_big_endian_int(images_file);
    
    LOG_INFO("MNIST: %u images de %ux%u", num_images, rows, cols);
    
    // Ouvrir le fichier de labels
    FILE *labels_file = fopen(labels_path, "rb");
    if (!labels_file) {
        LOG_ERROR("Impossible d'ouvrir: %s", labels_path);
        fclose(images_file);
        return NULL;
    }
    
    // Lire l'en-tête des labels
    uint32_t magic_labels = read_big_endian_int(labels_file);
    if (magic_labels != 2049) {
        LOG_ERROR("Format IDX invalide pour les labels");
        fclose(images_file);
        fclose(labels_file);
        return NULL;
    }
    
    uint32_t num_labels = read_big_endian_int(labels_file);
    
    if (num_images != num_labels) {
        LOG_ERROR("Nombre d'images != nombre de labels");
        fclose(images_file);
        fclose(labels_file);
        return NULL;
    }
    
    // Allouer le dataset
    MNISTDataset *dataset = (MNISTDataset*)malloc(sizeof(MNISTDataset));
    dataset->count = num_images;
    dataset->image_size = rows * cols;
    dataset->images = (float**)malloc(num_images * sizeof(float*));
    dataset->labels = (uint8_t*)malloc(num_images * sizeof(uint8_t));
    
    // Lire les images et labels
    uint8_t *pixel_buffer = (uint8_t*)malloc(rows * cols);
    
    for (size_t i = 0; i < num_images; i++) {
        // Lire les pixels
        fread(pixel_buffer, 1, rows * cols, images_file);
        
        // Allouer et normaliser l'image
        dataset->images[i] = (float*)malloc(rows * cols * sizeof(float));
        for (size_t j = 0; j < rows * cols; j++) {
            dataset->images[i][j] = pixel_buffer[j] / 255.0f;
        }
        
        // Lire le label
        fread(&dataset->labels[i], 1, 1, labels_file);
        
        if ((i + 1) % 10000 == 0) {
            LOG_INFO("Chargé %zu/%zu images...", i + 1, num_images);
        }
    }
    
    free(pixel_buffer);
    fclose(images_file);
    fclose(labels_file);
    
    LOG_INFO("Dataset MNIST chargé: %zu images", dataset->count);
    return dataset;
}

void free_mnist_dataset(MNISTDataset *dataset) {
    if (!dataset) return;
    
    for (size_t i = 0; i < dataset->count; i++) {
        free(dataset->images[i]);
    }
    free(dataset->images);
    free(dataset->labels);
    free(dataset);
}

// ============================================================================
// AUGMENTATION DE DONNÉES
// ============================================================================

float* augment_image(const float *image, int width, int height, 
                     float rotation, float translation, float noise_level) {
    float *augmented = (float*)malloc(width * height * sizeof(float));
    memcpy(augmented, image, width * height * sizeof(float));
    
    // Rotation simple (approximation)
    float angle = randf(-rotation, rotation) * M_PI / 180.0f;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);
    
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    
    float *temp = (float*)calloc(width * height, sizeof(float));
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Rotation autour du centre
            float dx = x - cx;
            float dy = y - cy;
            
            int src_x = (int)(dx * cos_a - dy * sin_a + cx);
            int src_y = (int)(dx * sin_a + dy * cos_a + cy);
            
            // Translation
            src_x += (int)randf(-translation, translation);
            src_y += (int)randf(-translation, translation);
            
            if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                temp[y * width + x] = image[src_y * width + src_x];
            }
        }
    }
    
    // Ajout de bruit gaussien
    for (int i = 0; i < width * height; i++) {
        float noise = randf(-noise_level, noise_level);
        augmented[i] = clamp(temp[i] + noise, 0.0f, 1.0f);
    }
    
    free(temp);
    return augmented;
}

// ============================================================================
// BATCHING
// ============================================================================

int** create_batches(size_t total_samples, size_t batch_size, int *num_batches) {
    *num_batches = (total_samples + batch_size - 1) / batch_size;
    
    int **batches = (int**)malloc(*num_batches * sizeof(int*));
    
    // Créer un tableau d'indices
    int *indices = (int*)malloc(total_samples * sizeof(int));
    for (size_t i = 0; i < total_samples; i++) {
        indices[i] = i;
    }
    
    // Mélanger
    shuffle_indices(indices, total_samples);
    
    // Créer les batches
    for (int b = 0; b < *num_batches; b++) {
        size_t start = b * batch_size;
        size_t end = min_int(start + batch_size, total_samples);
        size_t current_batch_size = end - start;
        
        batches[b] = (int*)malloc(current_batch_size * sizeof(int));
        memcpy(batches[b], indices + start, current_batch_size * sizeof(int));
    }
    
    free(indices);
    return batches;
}

void shuffle_dataset(MNISTDataset *dataset) {
    for (size_t i = dataset->count - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        
        // Échanger les images
        float *temp_img = dataset->images[i];
        dataset->images[i] = dataset->images[j];
        dataset->images[j] = temp_img;
        
        // Échanger les labels
        uint8_t temp_label = dataset->labels[i];
        dataset->labels[i] = dataset->labels[j];
        dataset->labels[j] = temp_label;
    }
}
