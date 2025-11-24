#ifndef CNN_TRAINING_H
#define CNN_TRAINING_H

#include "cnn_model.h"
#include "dataset_loader.h"

// ============================================================================
// OPTIMISEUR
// ============================================================================

typedef struct {
    float learning_rate;
    float momentum;
    float beta1;            // Pour Adam
    float beta2;            // Pour Adam
    float epsilon;          // Pour Adam
    int timestep;           // Pour Adam
} Optimizer;

// ============================================================================
// BACKWARD PASS
// ============================================================================

// Backward pass pour une couche de convolution
void conv_backward(ConvLayer *layer, const float *grad_output);

// Backward pass pour une couche de pooling
float* pool_backward(PoolLayer *layer, const float *grad_output);

// Backward pass pour une couche dense
float* dense_backward(DenseLayer *layer, const float *grad_output, bool had_relu);

// Backward pass complet (calcule tous les gradients)
void cnn_backward(CNNModel *model, const float *input, const float *target);

// ============================================================================
// MISE À JOUR DES POIDS
// ============================================================================

// Update avec SGD simple
void update_weights_sgd(CNNModel *model, float learning_rate);

// Update avec momentum
void update_weights_momentum(CNNModel *model, Optimizer *opt);

// Update avec Adam
void update_weights_adam(CNNModel *model, Optimizer *opt);

// ============================================================================
// ENTRAÎNEMENT
// ============================================================================

// Entraîne le modèle sur un dataset
// Returns: précision finale sur le dataset de validation
float train_cnn(CNNModel *model, MNISTDataset *train_data, MNISTDataset *val_data,
                int epochs, int batch_size, float learning_rate);

// Évalue le modèle sur un dataset
float evaluate_cnn(CNNModel *model, MNISTDataset *dataset);

// ============================================================================
// UTILITAIRES
// ============================================================================

// Calcule la loss pour un batch
float compute_loss(CNNModel *model, float **inputs, uint8_t *labels, int batch_size);

// Crée un optimiseur
Optimizer* create_optimizer(float learning_rate, float momentum);

#endif // CNN_TRAINING_H
