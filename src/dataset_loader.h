#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include "utils.h"
#include <stdint.h>

// ============================================================================
// STRUCTURES
// ============================================================================

// Dataset MNIST
typedef struct {
    float **images;        // Tableau d'images (chaque image = tableau de 784 floats)
    uint8_t *labels;       // Labels (0-9)
    size_t count;          // Nombre d'images
    size_t image_size;     // 784 pour MNIST (28x28)
} MNISTDataset;

// ============================================================================
// CHARGEMENT MNIST (FORMAT IDX)
// ============================================================================

// Charge le dataset MNIST depuis les fichiers IDX
// images_path: fichier d'images (ex: "train-images-idx3-ubyte")
// labels_path: fichier de labels (ex: "train-labels-idx1-ubyte")
MNISTDataset* load_mnist_dataset(const char *images_path, const char *labels_path);

// Libère la mémoire d'un dataset
void free_mnist_dataset(MNISTDataset *dataset);

// ============================================================================
// AUGMENTATION DE DONNÉES
// ============================================================================

// Applique des transformations aléatoires pour augmenter le dataset
// rotation: angle max de rotation (en degrés)
// translation: décalage max (en pixels)
// noise_level: niveau de bruit (0.0 - 0.1)
float* augment_image(const float *image, int width, int height, 
                     float rotation, float translation, float noise_level);

// ============================================================================
// BATCHING
// ============================================================================

// Crée des batches mélangés pour l'entraînement
// indices: tableau des indices à utiliser
// batch_size: taille des batches
// num_batches: (sortie) nombre de batches créés
int** create_batches(size_t total_samples, size_t batch_size, int *num_batches);

// Mélange un dataset (shuffle)
void shuffle_dataset(MNISTDataset *dataset);

#endif // DATASET_LOADER_H
