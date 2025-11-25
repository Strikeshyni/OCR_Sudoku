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
    
    // Allouer le dataset (allocation initiale max, on réduira après)
    MNISTDataset *dataset = (MNISTDataset*)malloc(sizeof(MNISTDataset));
    dataset->image_size = rows * cols;
    dataset->images = (float**)malloc(num_images * sizeof(float*));
    dataset->labels = (uint8_t*)malloc(num_images * sizeof(uint8_t));
    
    // Lire les images et labels
    uint8_t *pixel_buffer = (uint8_t*)malloc(rows * cols);
    size_t valid_count = 0;
    
    for (size_t i = 0; i < num_images; i++) {
        // Lire le label d'abord (pour savoir si on garde)
        // Attention: dans les fichiers IDX, les images sont stockées avant ou après ?
        // IDX images et labels sont dans des fichiers séparés.
        // On doit lire les deux en parallèle.
        
        uint8_t label;
        fread(&label, 1, 1, labels_file);
        
        // Lire les pixels
        fread(pixel_buffer, 1, rows * cols, images_file);
        
        // Filtrer le 0 (MNIST 0 ressemble à un zéro, pas à une case vide)
        if (label == 0) {
            continue; 
        }
        
        // Allouer et normaliser l'image
        dataset->images[valid_count] = (float*)malloc(rows * cols * sizeof(float));
        for (size_t j = 0; j < rows * cols; j++) {
            dataset->images[valid_count][j] = pixel_buffer[j] / 255.0f;
        }
        
        dataset->labels[valid_count] = label;
        valid_count++;
        
        if ((i + 1) % 10000 == 0) {
            LOG_INFO("  - Traité %zu images (gardé %zu)", i + 1, valid_count);
        }
    }
    
    // Réduire la taille des tableaux
    dataset->count = valid_count;
    // On pourrait realloc ici pour gagner de la place, mais c'est optionnel
    
    free(pixel_buffer);
    fclose(images_file);
    fclose(labels_file);
    
    LOG_INFO("Chargement terminé. %zu images conservées (0 filtrés).", dataset->count);
    
    return dataset;
}

// ============================================================================
// CHARGEMENT DATASET SUPPLÉMENTAIRE
// ============================================================================

void load_extra_dataset(const char *filepath, MNISTDataset *dataset) {
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        LOG_INFO("Fichier de données supplémentaires non trouvé: %s (ignoré)", filepath);
        return;
    }
    
    // Lire le header
    uint32_t magic = read_big_endian_int(file);
    if (magic != 0xDEADBEEF) {
        LOG_ERROR("Magic number invalide pour %s: 0x%X (attendu 0xDEADBEEF)", filepath, magic);
        fclose(file);
        return;
    }
    
    uint32_t count = read_big_endian_int(file);
    uint32_t width = read_big_endian_int(file);
    uint32_t height = read_big_endian_int(file);
    
    if (width * height != dataset->image_size) {
        LOG_ERROR("Dimensions incompatibles: %ux%u vs %zu (taille attendue)", 
                  width, height, dataset->image_size);
        fclose(file);
        return;
    }
    
    LOG_INFO("Chargement de %u images supplémentaires depuis %s...", count, filepath);
    
    // Réallouer la mémoire (estimation max)
    size_t max_new_count = dataset->count + count;
    float **new_images = (float**)realloc(dataset->images, max_new_count * sizeof(float*));
    uint8_t *new_labels = (uint8_t*)realloc(dataset->labels, max_new_count * sizeof(uint8_t));
    
    if (!new_images || !new_labels) {
        LOG_ERROR("Échec de la réallocation mémoire");
        fclose(file);
        return;
    }
    
    dataset->images = new_images;
    dataset->labels = new_labels;
    
    // Lire les données
    uint8_t *pixel_buffer = (uint8_t*)malloc(width * height);
    size_t added_count = 0;
    
    for (size_t i = 0; i < count; i++) {
        uint8_t label;
        fread(&label, 1, 1, file);
        fread(pixel_buffer, 1, width * height, file);
        
        if (label == 0) continue; // Filtrer les 0
        
        size_t idx = dataset->count + added_count;
        dataset->labels[idx] = label;
        
        // Allouer et normaliser
        dataset->images[idx] = (float*)malloc(width * height * sizeof(float));
        for (size_t j = 0; j < width * height; j++) {
            dataset->images[idx][j] = pixel_buffer[j] / 255.0f;
        }
        added_count++;
    }
    
    free(pixel_buffer);
    dataset->count += added_count;
    
    LOG_INFO("Ajouté %zu images (filtré %zu zéros). Total: %zu images.", 
             added_count, count - added_count, dataset->count);
    fclose(file);
}

// ============================================================================
// LIBÉRATION MÉMOIRE
// ============================================================================

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

// ============================================================================
// GÉNÉRATION DE CLASSE VIDE (0)
// ============================================================================

void generate_empty_samples(MNISTDataset *dataset, int count) {
    LOG_INFO("Génération de %d échantillons 'vides' (classe 0)...", count);
    
    size_t new_count = dataset->count + count;
    float **new_images = (float**)realloc(dataset->images, new_count * sizeof(float*));
    uint8_t *new_labels = (uint8_t*)realloc(dataset->labels, new_count * sizeof(uint8_t));
    
    if (!new_images || !new_labels) {
        LOG_ERROR("Échec réallocation pour empty samples");
        return;
    }
    
    dataset->images = new_images;
    dataset->labels = new_labels;
    
    for (int i = 0; i < count; i++) {
        size_t idx = dataset->count + i;
        dataset->labels[idx] = 0;
        dataset->images[idx] = (float*)malloc(dataset->image_size * sizeof(float));
        
        // Type de bruit
        float type = randf(0.0f, 1.0f);
        
        if (type < 0.7f) {
            // Cas 1: Presque noir (bruit très faible) - Cas le plus fréquent après seuillage
            for (size_t j = 0; j < dataset->image_size; j++) {
                dataset->images[idx][j] = randf(0.0f, 0.05f); 
            }
        } else if (type < 0.9f) {
            // Cas 2: Bruit uniforme un peu plus fort
            for (size_t j = 0; j < dataset->image_size; j++) {
                dataset->images[idx][j] = randf(0.0f, 0.15f);
            }
        } else {
            // Cas 3: Quelques artefacts (simulant des restes de bordures ou taches)
            for (size_t j = 0; j < dataset->image_size; j++) {
                dataset->images[idx][j] = randf(0.0f, 0.05f); // Fond noir
            }
            
            // Ajouter 1 à 3 "taches" ou lignes
            int num_spots = (int)randf(1, 4);
            for(int k=0; k<num_spots; k++) {
                int center = (int)randf(0, dataset->image_size);
                // Petit blob
                dataset->images[idx][center] = randf(0.5f, 1.0f);
                if (center + 1 < (int)dataset->image_size) dataset->images[idx][center+1] = randf(0.3f, 0.8f);
                if (center - 1 >= 0) dataset->images[idx][center-1] = randf(0.3f, 0.8f);
            }
        }
    }
    
    dataset->count = new_count;
    LOG_INFO("Ajouté %d images vides.", count);
}
