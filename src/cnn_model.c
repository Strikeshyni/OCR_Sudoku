#include "cnn_model.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// ============================================================================
// CRÉATION DES COUCHES
// ============================================================================

ConvLayer* create_conv_layer(int num_filters, int filter_size, int input_channels,
                              int input_width, int input_height) {
    ConvLayer *layer = (ConvLayer*)malloc(sizeof(ConvLayer));
    
    layer->num_filters = num_filters;
    layer->filter_size = filter_size;
    layer->input_channels = input_channels;
    layer->input_width = input_width;
    layer->input_height = input_height;
    layer->output_width = input_width - filter_size + 1;
    layer->output_height = input_height - filter_size + 1;
    
    // Initialisation des poids (Xavier/Glorot)
    int weight_count = num_filters * input_channels * filter_size * filter_size;
    layer->weights = (float*)malloc(weight_count * sizeof(float));
    float scale = sqrtf(2.0f / (input_channels * filter_size * filter_size));
    
    for (int i = 0; i < weight_count; i++) {
        layer->weights[i] = randf(-scale, scale);
    }
    
    layer->biases = (float*)calloc(num_filters, sizeof(float));
    
    // Caches pour backprop
    int input_size = input_channels * input_width * input_height;
    int output_size = num_filters * layer->output_width * layer->output_height;
    
    layer->input_cache = (float*)malloc(input_size * sizeof(float));
    layer->output_cache = (float*)malloc(output_size * sizeof(float));
    layer->weight_gradients = (float*)calloc(weight_count, sizeof(float));
    layer->bias_gradients = (float*)calloc(num_filters, sizeof(float));
    
    return layer;
}

PoolLayer* create_pool_layer(int pool_size, int input_channels,
                             int input_width, int input_height) {
    PoolLayer *layer = (PoolLayer*)malloc(sizeof(PoolLayer));
    
    layer->pool_size = pool_size;
    layer->input_channels = input_channels;
    layer->input_width = input_width;
    layer->input_height = input_height;
    layer->output_width = input_width / pool_size;
    layer->output_height = input_height / pool_size;
    
    int input_size = input_channels * input_width * input_height;
    int output_size = input_channels * layer->output_width * layer->output_height;
    
    layer->input_cache = (float*)malloc(input_size * sizeof(float));
    layer->max_indices = (int*)malloc(output_size * sizeof(int));
    
    return layer;
}

DenseLayer* create_dense_layer(int input_size, int output_size) {
    DenseLayer *layer = (DenseLayer*)malloc(sizeof(DenseLayer));
    
    layer->input_size = input_size;
    layer->output_size = output_size;
    
    // Initialisation des poids
    int weight_count = input_size * output_size;
    layer->weights = (float*)malloc(weight_count * sizeof(float));
    float scale = sqrtf(2.0f / input_size);
    
    for (int i = 0; i < weight_count; i++) {
        layer->weights[i] = randf(-scale, scale);
    }
    
    layer->biases = (float*)calloc(output_size, sizeof(float));
    
    layer->input_cache = (float*)malloc(input_size * sizeof(float));
    layer->output_cache = (float*)malloc(output_size * sizeof(float));
    layer->weight_gradients = (float*)calloc(weight_count, sizeof(float));
    layer->bias_gradients = (float*)calloc(output_size, sizeof(float));
    
    return layer;
}

CNNModel* create_cnn_model() {
    CNNModel *model = (CNNModel*)malloc(sizeof(CNNModel));
    
    // Architecture LeNet-5 adaptée pour MNIST
    // Input: 28x28x1
    model->conv1 = create_conv_layer(6, 5, 1, 28, 28);      // -> 24x24x6
    model->pool1 = create_pool_layer(2, 6, 24, 24);         // -> 12x12x6
    model->conv2 = create_conv_layer(16, 5, 6, 12, 12);     // -> 8x8x16
    model->pool2 = create_pool_layer(2, 16, 8, 8);          // -> 4x4x16 = 256
    model->fc1 = create_dense_layer(256, 120);              // -> 120
    model->fc2 = create_dense_layer(120, 10);               // -> 10 (classes)
    
    LOG_INFO("Modèle CNN créé: Conv(6,5x5)->Pool(2x2)->Conv(16,5x5)->Pool(2x2)->FC(120)->FC(10)");
    return model;
}

// ============================================================================
// LIBÉRATION
// ============================================================================

void free_conv_layer(ConvLayer *layer) {
    if (!layer) return;
    free(layer->weights);
    free(layer->biases);
    free(layer->input_cache);
    free(layer->output_cache);
    free(layer->weight_gradients);
    free(layer->bias_gradients);
    free(layer);
}

void free_pool_layer(PoolLayer *layer) {
    if (!layer) return;
    free(layer->input_cache);
    free(layer->max_indices);
    free(layer);
}

void free_dense_layer(DenseLayer *layer) {
    if (!layer) return;
    free(layer->weights);
    free(layer->biases);
    free(layer->input_cache);
    free(layer->output_cache);
    free(layer->weight_gradients);
    free(layer->bias_gradients);
    free(layer);
}

void free_cnn_model(CNNModel *model) {
    if (!model) return;
    free_conv_layer(model->conv1);
    free_pool_layer(model->pool1);
    free_conv_layer(model->conv2);
    free_pool_layer(model->pool2);
    free_dense_layer(model->fc1);
    free_dense_layer(model->fc2);
    free(model);
}

// ============================================================================
// FORWARD PASS - CONVOLUTION
// ============================================================================

float* conv_forward(ConvLayer *layer, const float *input) {
    // Sauvegarder l'entrée pour la backprop
    int input_size = layer->input_channels * layer->input_width * layer->input_height;
    memcpy(layer->input_cache, input, input_size * sizeof(float));
    
    int out_w = layer->output_width;
    int out_h = layer->output_height;
    int f_size = layer->filter_size;
    
    // Pour chaque filtre
    for (int f = 0; f < layer->num_filters; f++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                float sum = layer->biases[f];
                
                // Convolution sur tous les canaux d'entrée
                for (int c = 0; c < layer->input_channels; c++) {
                    for (int fy = 0; fy < f_size; fy++) {
                        for (int fx = 0; fx < f_size; fx++) {
                            int in_y = y + fy;
                            int in_x = x + fx;
                            
                            int input_idx = c * (layer->input_width * layer->input_height) +
                                          in_y * layer->input_width + in_x;
                            int weight_idx = f * (layer->input_channels * f_size * f_size) +
                                           c * (f_size * f_size) + fy * f_size + fx;
                            
                            sum += input[input_idx] * layer->weights[weight_idx];
                        }
                    }
                }
                
                // Activation ReLU
                int out_idx = f * (out_w * out_h) + y * out_w + x;
                layer->output_cache[out_idx] = relu(sum);
            }
        }
    }
    
    return layer->output_cache;
}

// ============================================================================
// FORWARD PASS - POOLING
// ============================================================================

float* pool_forward(PoolLayer *layer, const float *input) {
    int input_size = layer->input_channels * layer->input_width * layer->input_height;
    memcpy(layer->input_cache, input, input_size * sizeof(float));
    
    int p_size = layer->pool_size;
    int out_w = layer->output_width;
    int out_h = layer->output_height;
    
    float *output = (float*)malloc(layer->input_channels * out_w * out_h * sizeof(float));
    
    for (int c = 0; c < layer->input_channels; c++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                float max_val = -INFINITY;
                int max_idx = 0;
                
                // Trouver le maximum dans la fenêtre de pooling
                for (int py = 0; py < p_size; py++) {
                    for (int px = 0; px < p_size; px++) {
                        int in_y = y * p_size + py;
                        int in_x = x * p_size + px;
                        int in_idx = c * (layer->input_width * layer->input_height) +
                                    in_y * layer->input_width + in_x;
                        
                        if (input[in_idx] > max_val) {
                            max_val = input[in_idx];
                            max_idx = in_idx;
                        }
                    }
                }
                
                int out_idx = c * (out_w * out_h) + y * out_w + x;
                output[out_idx] = max_val;
                layer->max_indices[out_idx] = max_idx;
            }
        }
    }
    
    return output;
}

// ============================================================================
// FORWARD PASS - DENSE
// ============================================================================

float* dense_forward(DenseLayer *layer, const float *input, bool use_relu) {
    memcpy(layer->input_cache, input, layer->input_size * sizeof(float));
    
    for (int i = 0; i < layer->output_size; i++) {
        float sum = layer->biases[i];
        
        for (int j = 0; j < layer->input_size; j++) {
            sum += input[j] * layer->weights[i * layer->input_size + j];
        }
        
        layer->output_cache[i] = use_relu ? relu(sum) : sum;
    }
    
    return layer->output_cache;
}

// ============================================================================
// FORWARD PASS COMPLET
// ============================================================================

float* cnn_forward(CNNModel *model, const float *input) {
    // Conv1 -> Pool1
    float *out1 = conv_forward(model->conv1, input);
    float *pool1_out = pool_forward(model->pool1, out1);
    
    // Conv2 -> Pool2
    float *out2 = conv_forward(model->conv2, pool1_out);
    float *pool2_out = pool_forward(model->pool2, out2);
    
    free(pool1_out);
    
    // Aplatir pour les couches denses
    float *flattened = pool2_out;
    
    // FC1 avec ReLU
    float *fc1_out = dense_forward(model->fc1, flattened, true);
    
    // FC2 (sortie)
    float *logits = dense_forward(model->fc2, fc1_out, false);
    
    // Softmax
    float *probabilities = (float*)malloc(10 * sizeof(float));
    softmax(logits, probabilities, 10);
    
    return probabilities;
}

int cnn_predict(CNNModel *model, const float *input) {
    float *probs = cnn_forward(model, input);
    
    int predicted_class = 0;
    float max_prob = probs[0];
    
    for (int i = 1; i < 10; i++) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            predicted_class = i;
        }
    }
    
    free(probs);
    return predicted_class;
}

// ============================================================================
// SAUVEGARDE ET CHARGEMENT
// ============================================================================

bool save_cnn_weights(const CNNModel *model, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        LOG_ERROR("Impossible de sauvegarder les poids: %s", filename);
        return false;
    }
    
    // Magic number pour vérification
    uint32_t magic = 0x434E4E57;  // "CNNW"
    fwrite(&magic, sizeof(uint32_t), 1, file);
    
    // Conv1
    fwrite(model->conv1->weights, sizeof(float), 
           model->conv1->num_filters * model->conv1->input_channels * 
           model->conv1->filter_size * model->conv1->filter_size, file);
    fwrite(model->conv1->biases, sizeof(float), model->conv1->num_filters, file);
    
    // Conv2
    fwrite(model->conv2->weights, sizeof(float), 
           model->conv2->num_filters * model->conv2->input_channels * 
           model->conv2->filter_size * model->conv2->filter_size, file);
    fwrite(model->conv2->biases, sizeof(float), model->conv2->num_filters, file);
    
    // FC1
    fwrite(model->fc1->weights, sizeof(float), 
           model->fc1->input_size * model->fc1->output_size, file);
    fwrite(model->fc1->biases, sizeof(float), model->fc1->output_size, file);
    
    // FC2
    fwrite(model->fc2->weights, sizeof(float), 
           model->fc2->input_size * model->fc2->output_size, file);
    fwrite(model->fc2->biases, sizeof(float), model->fc2->output_size, file);
    
    fclose(file);
    LOG_INFO("Poids sauvegardés: %s", filename);
    return true;
}

bool load_cnn_weights(CNNModel *model, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        LOG_ERROR("Impossible de charger les poids: %s", filename);
        return false;
    }
    
    // Vérifier le magic number
    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, file);
    if (magic != 0x434E4E57) {
        LOG_ERROR("Format de fichier invalide");
        fclose(file);
        return false;
    }
    
    // Charger les poids (même ordre que la sauvegarde)
    fread(model->conv1->weights, sizeof(float), 
          model->conv1->num_filters * model->conv1->input_channels * 
          model->conv1->filter_size * model->conv1->filter_size, file);
    fread(model->conv1->biases, sizeof(float), model->conv1->num_filters, file);
    
    fread(model->conv2->weights, sizeof(float), 
          model->conv2->num_filters * model->conv2->input_channels * 
          model->conv2->filter_size * model->conv2->filter_size, file);
    fread(model->conv2->biases, sizeof(float), model->conv2->num_filters, file);
    
    fread(model->fc1->weights, sizeof(float), 
          model->fc1->input_size * model->fc1->output_size, file);
    fread(model->fc1->biases, sizeof(float), model->fc1->output_size, file);
    
    fread(model->fc2->weights, sizeof(float), 
          model->fc2->input_size * model->fc2->output_size, file);
    fread(model->fc2->biases, sizeof(float), model->fc2->output_size, file);
    
    fclose(file);
    LOG_INFO("Poids chargés: %s", filename);
    return true;
}
