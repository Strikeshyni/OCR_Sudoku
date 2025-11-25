#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cnn_model.h"
#include "dataset_loader.h"
#include "utils.h"

void print_ascii_art(float *image, int width, int height) {
    const char *chars = " .:-=+*#%@";
    printf("\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float val = image[y * width + x];
            int idx = (int)(val * 9.99f);
            if (idx < 0) idx = 0;
            if (idx > 9) idx = 9;
            printf("%c%c", chars[idx], chars[idx]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_confusion_matrix(int matrix[10][10]) {
    printf("\n=== Table de Vérité (Confusion Matrix) ===\n");
    printf("Lignes : Classe Réelle\n");
    printf("Colonnes : Classe Prédite\n\n");
    printf("      ");
    for (int i = 0; i < 10; i++) printf("%4d ", i);
    printf("\n");
    printf("      ");
    for (int i = 0; i < 10; i++) printf("---- ");
    printf("\n");
    
    for (int i = 0; i < 10; i++) {
        printf("%4d |", i);
        for (int j = 0; j < 10; j++) {
            printf("%4d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_metrics(int matrix[10][10]) {
    printf("\n=== Métriques par Classe ===\n");
    printf("Classe | Précision | Rappel    | F1-Score\n");
    printf("-------|-----------|-----------|----------\n");
    
    float total_f1 = 0;
    int classes_count = 0;
    
    for (int i = 0; i < 10; i++) {
        int tp = matrix[i][i];
        int fp = 0;
        int fn = 0;
        
        for (int j = 0; j < 10; j++) {
            if (i != j) {
                fp += matrix[j][i]; // Colonne i, ligne j (prédit i mais c'était j)
                fn += matrix[i][j]; // Ligne i, colonne j (était i mais prédit j)
            }
        }
        
        float precision = (tp + fp) > 0 ? (float)tp / (tp + fp) : 0;
        float recall = (tp + fn) > 0 ? (float)tp / (tp + fn) : 0;
        float f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
        
        char label_name[10];
        if (i == 0) snprintf(label_name, 10, "Vide");
        else snprintf(label_name, 10, "%d   ", i);
        
        printf("   %s   |   %5.1f%%  |   %5.1f%%  |   %5.1f%%\n", 
               label_name, precision * 100, recall * 100, f1 * 100);
               
        total_f1 += f1;
        classes_count++;
    }
    
    printf("\nF1-Score Moyen: %.1f%%\n", (total_f1 / classes_count) * 100);
}

int main() {
    // 1. Charger les données de test MNIST
    LOG_INFO("Chargement des données de test MNIST...");
    MNISTDataset *dataset = load_mnist_dataset("data/mnist/t10k-images.idx3-ubyte", "data/mnist/t10k-labels.idx1-ubyte");
    if (!dataset) {
        LOG_ERROR("Impossible de charger MNIST test.");
        return 1;
    }
    
    // 2. Charger les données de test Digital (si disponibles)
    LOG_INFO("Chargement des données de test Digital...");
    load_extra_dataset("data/digital_test.bin", dataset);
    
    // 2b. Générer des échantillons vides (classe 0)
    LOG_INFO("Génération de la classe 'Vide' (0)...");
    int empty_count = dataset->count / 9;
    generate_empty_samples(dataset, empty_count);
    
    // 3. Charger le modèle
    LOG_INFO("Chargement du modèle...");
    CNNModel *model = create_cnn_model();
    if (!load_cnn_weights(model, "models/cnn_weights_best.bin")) {
        LOG_INFO("Poids 'cnn_weights_best.bin' non trouvés, essai avec 'cnn_weights.bin'...");
        if (!load_cnn_weights(model, "models/cnn_weights.bin")) {
             LOG_ERROR("Impossible de charger les poids du modèle.");
             free_mnist_dataset(dataset);
             free_cnn_model(model);
             return 1;
        }
    }
    
    // 4. Évaluation
    int confusion_matrix[10][10] = {0};
    int correct = 0;
    
    LOG_INFO("Évaluation sur %zu images...", dataset->count);
    
    // Indices pour afficher quelques exemples (ex: 5 premiers de MNIST et 5 premiers de Digital)
    // MNIST est au début, Digital est à la fin (après 10000)
    size_t examples_to_show[] = {0, 1, 2, 3, 4, 10000, 10001, 10002, 10003, 10004}; 
    size_t num_examples = 10;
    
    for (size_t i = 0; i < dataset->count; i++) {
        int prediction = cnn_predict(model, dataset->images[i]);
        int actual = dataset->labels[i];
        
        confusion_matrix[actual][prediction]++;
        if (prediction == actual) correct++;
        
        // Afficher quelques exemples
        bool show = false;
        for(size_t k=0; k<num_examples; k++) {
            if(i == examples_to_show[k] && i < dataset->count) show = true;
        }
        
        if (show) {
             printf("\n--- Exemple Image #%zu ---\n", i);
             print_ascii_art(dataset->images[i], 28, 28);
             printf("Label Réel: %d, Prédiction: %d [%s]\n", 
                    actual, prediction, (prediction==actual) ? "CORRECT" : "ERREUR");
        }
    }
    
    // 5. Résultats
    printf("\n==================================================\n");
    printf("RÉSULTATS GLOBAUX\n");
    printf("==================================================\n");
    printf("Images testées: %zu\n", dataset->count);
    printf("Correctes:      %d\n", correct);
    printf("Précision:      %.2f%%\n", (float)correct / dataset->count * 100.0f);
    
    print_confusion_matrix(confusion_matrix);
    print_metrics(confusion_matrix);
    
    // Cleanup
    free_cnn_model(model);
    free_mnist_dataset(dataset);
    
    return 0;
}
