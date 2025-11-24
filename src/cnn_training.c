#include "cnn_training.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// BACKWARD PASS - DENSE
// ============================================================================

float* dense_backward(DenseLayer *layer, const float *grad_output, bool had_relu) {
    float *grad_input = (float*)calloc(layer->input_size, sizeof(float));
    
    // Appliquer la dérivée de ReLU si nécessaire
    float *grad_activated = (float*)malloc(layer->output_size * sizeof(float));
    for (int i = 0; i < layer->output_size; i++) {
        if (had_relu) {
            grad_activated[i] = grad_output[i] * relu_derivative(layer->output_cache[i]);
        } else {
            grad_activated[i] = grad_output[i];
        }
    }
    
    // Gradients des poids et biais
    for (int i = 0; i < layer->output_size; i++) {
        layer->bias_gradients[i] += grad_activated[i];
        
        for (int j = 0; j < layer->input_size; j++) {
            layer->weight_gradients[i * layer->input_size + j] += 
                grad_activated[i] * layer->input_cache[j];
        }
    }
    
    // Gradient par rapport à l'entrée
    for (int j = 0; j < layer->input_size; j++) {
        for (int i = 0; i < layer->output_size; i++) {
            grad_input[j] += grad_activated[i] * layer->weights[i * layer->input_size + j];
        }
    }
    
    free(grad_activated);
    return grad_input;
}

// ============================================================================
// BACKWARD PASS - POOLING
// ============================================================================

float* pool_backward(PoolLayer *layer, const float *grad_output) {
    int input_size = layer->input_channels * layer->input_width * layer->input_height;
    float *grad_input = (float*)calloc(input_size, sizeof(float));
    
    int out_w = layer->output_width;
    int out_h = layer->output_height;
    
    // Propager le gradient seulement vers les positions qui étaient maximales
    for (int c = 0; c < layer->input_channels; c++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                int out_idx = c * (out_w * out_h) + y * out_w + x;
                int max_idx = layer->max_indices[out_idx];
                grad_input[max_idx] += grad_output[out_idx];
            }
        }
    }
    
    return grad_input;
}

// ============================================================================
// BACKWARD PASS - CONVOLUTION
// ============================================================================

void conv_backward(ConvLayer *layer, const float *grad_output) {
    int out_w = layer->output_width;
    int out_h = layer->output_height;
    int f_size = layer->filter_size;
    
    // Pour chaque filtre
    for (int f = 0; f < layer->num_filters; f++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                int out_idx = f * (out_w * out_h) + y * out_w + x;
                
                // Gradient de ReLU
                float grad = grad_output[out_idx];
                if (layer->output_cache[out_idx] <= 0) grad = 0;
                
                // Gradient du biais
                layer->bias_gradients[f] += grad;
                
                // Gradients des poids
                for (int c = 0; c < layer->input_channels; c++) {
                    for (int fy = 0; fy < f_size; fy++) {
                        for (int fx = 0; fx < f_size; fx++) {
                            int in_y = y + fy;
                            int in_x = x + fx;
                            
                            int input_idx = c * (layer->input_width * layer->input_height) +
                                          in_y * layer->input_width + in_x;
                            int weight_idx = f * (layer->input_channels * f_size * f_size) +
                                           c * (f_size * f_size) + fy * f_size + fx;
                            
                            layer->weight_gradients[weight_idx] += 
                                grad * layer->input_cache[input_idx];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// BACKWARD PASS COMPLET
// ============================================================================

void cnn_backward(CNNModel *model, const float *input, const float *target) {
    // Forward pass pour remplir les caches
    float *output = cnn_forward(model, input);
    
    // Gradient de la loss (cross-entropy + softmax)
    float *grad_output = (float*)malloc(10 * sizeof(float));
    for (int i = 0; i < 10; i++) {
        grad_output[i] = output[i] - target[i];
    }
    
    // Backward FC2
    float *grad_fc2 = dense_backward(model->fc2, grad_output, false);
    
    // Backward FC1
    float *grad_fc1 = dense_backward(model->fc1, grad_fc2, true);
    
    // Reshape pour Pool2
    float *grad_pool2 = grad_fc1;  // Même taille (256)
    
    // Backward Pool2
    float *grad_conv2_out = pool_backward(model->pool2, grad_pool2);
    
    // Backward Conv2
    conv_backward(model->conv2, grad_conv2_out);
    
    // Backward Pool1 (besoin du gradient par rapport à conv2 input)
    // Simplification: on propage à travers conv2 de manière approximative
    // Dans une impl complète, il faudrait calculer le gradient complet
    // Pour l'instant, on s'arrête ici car conv1 et pool1 sont déjà entraînés
    
    free(output);
    free(grad_output);
    free(grad_fc2);
    free(grad_conv2_out);
}

// ============================================================================
// MISE À JOUR DES POIDS
// ============================================================================

static void update_layer_weights_sgd(float *weights, float *gradients, int count, float lr) {
    for (int i = 0; i < count; i++) {
        weights[i] -= lr * gradients[i];
        gradients[i] = 0;  // Reset gradient
    }
}

void update_weights_sgd(CNNModel *model, float learning_rate) {
    // Conv1
    int conv1_w_count = model->conv1->num_filters * model->conv1->input_channels *
                        model->conv1->filter_size * model->conv1->filter_size;
    update_layer_weights_sgd(model->conv1->weights, model->conv1->weight_gradients,
                            conv1_w_count, learning_rate);
    update_layer_weights_sgd(model->conv1->biases, model->conv1->bias_gradients,
                            model->conv1->num_filters, learning_rate);
    
    // Conv2
    int conv2_w_count = model->conv2->num_filters * model->conv2->input_channels *
                        model->conv2->filter_size * model->conv2->filter_size;
    update_layer_weights_sgd(model->conv2->weights, model->conv2->weight_gradients,
                            conv2_w_count, learning_rate);
    update_layer_weights_sgd(model->conv2->biases, model->conv2->bias_gradients,
                            model->conv2->num_filters, learning_rate);
    
    // FC1
    int fc1_w_count = model->fc1->input_size * model->fc1->output_size;
    update_layer_weights_sgd(model->fc1->weights, model->fc1->weight_gradients,
                            fc1_w_count, learning_rate);
    update_layer_weights_sgd(model->fc1->biases, model->fc1->bias_gradients,
                            model->fc1->output_size, learning_rate);
    
    // FC2
    int fc2_w_count = model->fc2->input_size * model->fc2->output_size;
    update_layer_weights_sgd(model->fc2->weights, model->fc2->weight_gradients,
                            fc2_w_count, learning_rate);
    update_layer_weights_sgd(model->fc2->biases, model->fc2->bias_gradients,
                            model->fc2->output_size, learning_rate);
}

// ============================================================================
// ENTRAÎNEMENT
// ============================================================================

float train_cnn(CNNModel *model, MNISTDataset *train_data, MNISTDataset *val_data,
                int epochs, int batch_size, float learning_rate) {
    
    LOG_INFO("Début de l'entraînement: %d époques, batch_size=%d, lr=%.4f", 
             epochs, batch_size, learning_rate);
    
    float best_val_acc = 0.0f;
    int patience = 5;
    int epochs_no_improve = 0;
    float min_delta = 0.001f; // 0.1% improvement required
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle_dataset(train_data);
        
        float epoch_loss = 0.0f;
        int num_batches = (train_data->count + batch_size - 1) / batch_size;
        
        for (int b = 0; b < num_batches; b++) {
            int start = b * batch_size;
            int end = (start + batch_size < (int)train_data->count) ? 
                      start + batch_size : train_data->count;
            int current_batch_size = end - start;
            
            // Forward + backward pour chaque exemple du batch
            for (int i = start; i < end; i++) {
                // Créer le vecteur one-hot du label
                float target[10] = {0};
                target[train_data->labels[i]] = 1.0f;
                
                // Backward pass (accumule les gradients)
                cnn_backward(model, train_data->images[i], target);
                
                // Calculer la loss (pour le suivi)
                float *output = cnn_forward(model, train_data->images[i]);
                epoch_loss += cross_entropy_loss(output, target, 10);
                free(output);
            }
            
            // Mise à jour des poids (moyenne des gradients du batch)
            float batch_lr = learning_rate / current_batch_size;
            update_weights_sgd(model, batch_lr);
            
            if ((b + 1) % 100 == 0) {
                LOG_INFO("Epoch %d/%d - Batch %d/%d", 
                         epoch + 1, epochs, b + 1, num_batches);
            }
        }
        
        epoch_loss /= train_data->count;
        
        // Évaluation sur le dataset de validation
        float val_acc = evaluate_cnn(model, val_data);
        
        LOG_INFO("Epoch %d/%d - Loss: %.4f - Val Acc: %.2f%%", 
                 epoch + 1, epochs, epoch_loss, val_acc * 100);
        
        if (val_acc > best_val_acc + min_delta) {
            best_val_acc = val_acc;
            epochs_no_improve = 0;
            save_cnn_weights(model, "models/cnn_weights_best.bin");
            LOG_INFO("Nouveau meilleur modèle sauvegardé!");
        } else {
            epochs_no_improve++;
            LOG_INFO("Pas d'amélioration depuis %d époques", epochs_no_improve);
        }

        if (epochs_no_improve >= patience) {
            LOG_INFO("Arrêt précoce (Early Stopping) déclenché!");
            break;
        }
        
        // Sauvegarder régulièrement
        if ((epoch + 1) % 5 == 0) {
            save_cnn_weights(model, "models/cnn_weights.bin");
        }
    }
    
    // Restore best weights
    LOG_INFO("Restauration des meilleurs poids...");
    load_cnn_weights(model, "models/cnn_weights_best.bin");
    
    LOG_INFO("Entraînement terminé. Meilleure précision: %.2f%%", best_val_acc * 100);
    return best_val_acc;
}

// ============================================================================
// ÉVALUATION
// ============================================================================

float evaluate_cnn(CNNModel *model, MNISTDataset *dataset) {
    int correct = 0;
    
    for (size_t i = 0; i < dataset->count; i++) {
        int predicted = cnn_predict(model, dataset->images[i]);
        if (predicted == dataset->labels[i]) {
            correct++;
        }
        
        if ((i + 1) % 1000 == 0) {
            LOG_DEBUG("Évaluation: %zu/%zu", i + 1, dataset->count);
        }
    }
    
    float accuracy = (float)correct / dataset->count;
    return accuracy;
}

// ============================================================================
// UTILITAIRES
// ============================================================================

float compute_loss(CNNModel *model, float **inputs, uint8_t *labels, int batch_size) {
    float total_loss = 0.0f;
    
    for (int i = 0; i < batch_size; i++) {
        float target[10] = {0};
        target[labels[i]] = 1.0f;
        
        float *output = cnn_forward(model, inputs[i]);
        total_loss += cross_entropy_loss(output, target, 10);
        free(output);
    }
    
    return total_loss / batch_size;
}

Optimizer* create_optimizer(float learning_rate, float momentum) {
    Optimizer *opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->learning_rate = learning_rate;
    opt->momentum = momentum;
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->epsilon = 1e-8f;
    opt->timestep = 0;
    return opt;
}
