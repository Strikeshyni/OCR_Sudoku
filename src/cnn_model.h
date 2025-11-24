#ifndef CNN_MODEL_H
#define CNN_MODEL_H

#include "utils.h"
#include <stdint.h>

// ============================================================================
// ARCHITECTURE CNN (LeNet-style pour MNIST)
// ============================================================================

// Couche de convolution
typedef struct {
    int num_filters;        // Nombre de filtres (kernels)
    int filter_size;        // Taille du filtre (ex: 5 pour 5x5)
    int input_channels;     // Canaux d'entrée
    int input_width;
    int input_height;
    int output_width;
    int output_height;
    
    float *weights;         // Poids des filtres
    float *biases;          // Biais (un par filtre)
    
    // Pour la backpropagation
    float *input_cache;     // Cache de l'entrée
    float *output_cache;    // Cache de la sortie
    float *weight_gradients;
    float *bias_gradients;
} ConvLayer;

// Couche de pooling (max pooling)
typedef struct {
    int pool_size;          // Taille du pooling (ex: 2 pour 2x2)
    int input_channels;
    int input_width;
    int input_height;
    int output_width;
    int output_height;
    
    // Pour la backpropagation
    float *input_cache;
    int *max_indices;       // Indices des maxima pour le backward pass
} PoolLayer;

// Couche dense (fully connected)
typedef struct {
    int input_size;
    int output_size;
    
    float *weights;         // Matrice de poids
    float *biases;          // Vecteur de biais
    
    // Pour la backpropagation
    float *input_cache;
    float *output_cache;
    float *weight_gradients;
    float *bias_gradients;
} DenseLayer;

// Modèle CNN complet
typedef struct {
    ConvLayer *conv1;       // 1ère couche conv
    PoolLayer *pool1;       // 1er max pooling
    ConvLayer *conv2;       // 2ème couche conv
    PoolLayer *pool2;       // 2ème max pooling
    DenseLayer *fc1;        // Couche dense 1
    DenseLayer *fc2;        // Couche dense 2 (sortie)
} CNNModel;

// ============================================================================
// CRÉATION ET LIBÉRATION
// ============================================================================

ConvLayer* create_conv_layer(int num_filters, int filter_size, int input_channels,
                              int input_width, int input_height);
PoolLayer* create_pool_layer(int pool_size, int input_channels,
                             int input_width, int input_height);
DenseLayer* create_dense_layer(int input_size, int output_size);

void free_conv_layer(ConvLayer *layer);
void free_pool_layer(PoolLayer *layer);
void free_dense_layer(DenseLayer *layer);

// Crée le modèle CNN complet pour MNIST (10 classes)
CNNModel* create_cnn_model();
void free_cnn_model(CNNModel *model);

// ============================================================================
// FORWARD PASS
// ============================================================================

// Forward pass d'une couche de convolution
float* conv_forward(ConvLayer *layer, const float *input);

// Forward pass d'une couche de pooling
float* pool_forward(PoolLayer *layer, const float *input);

// Forward pass d'une couche dense
float* dense_forward(DenseLayer *layer, const float *input, bool use_relu);

// Forward pass complet du modèle (retourne les probabilités softmax)
float* cnn_forward(CNNModel *model, const float *input);

// Prédiction (retourne la classe prédite 0-9)
int cnn_predict(CNNModel *model, const float *input);

// ============================================================================
// SAUVEGARDE ET CHARGEMENT
// ============================================================================

bool save_cnn_weights(const CNNModel *model, const char *filename);
bool load_cnn_weights(CNNModel *model, const char *filename);

#endif // CNN_MODEL_H
