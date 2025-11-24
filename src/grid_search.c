#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "utils.h"
#include "cnn_model.h"
#include "cnn_training.h"
#include "dataset_loader.h"

// Structure pour stocker les résultats d'une configuration
typedef struct {
    // Hyperparamètres
    int epochs;
    int batch_size;
    float learning_rate;
    float momentum;
    
    // Métriques globales
    float accuracy;
    float avg_f1_score;
    
    // Métriques par classe (digits 0-9)
    float precision[10];
    float recall[10];
    float f1_score[10];
    
    // Matrice de confusion 10x10
    int confusion_matrix[10][10];
    
    // Temps d'entraînement
    double training_time;
} GridSearchResult;

// Calculer les métriques détaillées pour chaque classe
void compute_metrics(CNNModel *model, MNISTDataset *dataset, GridSearchResult *result) {
    // Réinitialiser la matrice de confusion
    memset(result->confusion_matrix, 0, sizeof(result->confusion_matrix));
    
    // Remplir la matrice de confusion
    int correct = 0;
    for (size_t i = 0; i < dataset->count; i++) {
        int predicted = cnn_predict(model, dataset->images[i]);
        int actual = dataset->labels[i];
        
        result->confusion_matrix[actual][predicted]++;
        if (predicted == actual) correct++;
    }
    
    result->accuracy = (float)correct / dataset->count;
    
    // Calculer précision, recall et F1-score pour chaque classe
    float total_f1 = 0.0f;
    for (int digit = 0; digit < 10; digit++) {
        // True Positives: confusion_matrix[digit][digit]
        int tp = result->confusion_matrix[digit][digit];
        
        // False Positives: somme de la colonne 'digit' sauf tp
        int fp = 0;
        for (int i = 0; i < 10; i++) {
            if (i != digit) fp += result->confusion_matrix[i][digit];
        }
        
        // False Negatives: somme de la ligne 'digit' sauf tp
        int fn = 0;
        for (int j = 0; j < 10; j++) {
            if (j != digit) fn += result->confusion_matrix[digit][j];
        }
        
        // Calcul des métriques
        result->precision[digit] = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.0f;
        result->recall[digit] = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.0f;
        
        if (result->precision[digit] + result->recall[digit] > 0) {
            result->f1_score[digit] = 2 * result->precision[digit] * result->recall[digit] 
                                     / (result->precision[digit] + result->recall[digit]);
        } else {
            result->f1_score[digit] = 0.0f;
        }
        
        total_f1 += result->f1_score[digit];
    }
    
    result->avg_f1_score = total_f1 / 10.0f;
}

// Afficher un résumé des métriques
void print_metrics_summary(GridSearchResult *result) {
    printf("\n┌─────────────────────────────────────────────────────────┐\n");
    printf("│ Hyperparamètres                                         │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ Epochs:         %-3d                                     │\n", result->epochs);
    printf("│ Batch size:     %-3d                                     │\n", result->batch_size);
    printf("│ Learning rate:  %.4f                                  │\n", result->learning_rate);
    printf("│ Momentum:       %.2f                                    │\n", result->momentum);
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ Métriques Globales                                      │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│ Accuracy:       %.4f (%.2f%%)                         │\n", 
           result->accuracy, result->accuracy * 100);
    printf("│ Avg F1-Score:   %.4f                                  │\n", 
           result->avg_f1_score);
    printf("│ Training time:  %.2f min                              │\n", 
           result->training_time / 60.0);
    printf("└─────────────────────────────────────────────────────────┘\n");
}

// Afficher le tableau détaillé par classe
void print_per_class_metrics(GridSearchResult *result) {
    printf("\n┌───────┬───────────┬──────────┬──────────┐\n");
    printf("│ Digit │ Precision │  Recall  │ F1-Score │\n");
    printf("├───────┼───────────┼──────────┼──────────┤\n");
    
    for (int i = 0; i < 10; i++) {
        printf("│   %d   │  %.4f   │  %.4f  │  %.4f  │\n",
               i, result->precision[i], result->recall[i], result->f1_score[i]);
    }
    
    printf("└───────┴───────────┴──────────┴──────────┘\n");
}

// Sauvegarder les résultats dans un fichier CSV
void save_results_to_csv(GridSearchResult *results, int count, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        LOG_ERROR("Impossible d'ouvrir %s pour écriture", filename);
        return;
    }
    
    // En-tête
    fprintf(f, "epochs,batch_size,learning_rate,momentum,accuracy,avg_f1_score,training_time");
    for (int i = 0; i < 10; i++) {
        fprintf(f, ",precision_%d,recall_%d,f1_%d", i, i, i);
    }
    fprintf(f, "\n");
    
    // Données
    for (int r = 0; r < count; r++) {
        GridSearchResult *res = &results[r];
        fprintf(f, "%d,%d,%.4f,%.2f,%.4f,%.4f,%.2f",
                res->epochs, res->batch_size, res->learning_rate, res->momentum,
                res->accuracy, res->avg_f1_score, res->training_time);
        
        for (int i = 0; i < 10; i++) {
            fprintf(f, ",%.4f,%.4f,%.4f", 
                    res->precision[i], res->recall[i], res->f1_score[i]);
        }
        fprintf(f, "\n");
    }
    
    fclose(f);
    LOG_INFO("Résultats sauvegardés dans %s", filename);
}

// Comparer deux résultats (pour trier)
int compare_results(const void *a, const void *b) {
    const GridSearchResult *ra = (const GridSearchResult*)a;
    const GridSearchResult *rb = (const GridSearchResult*)b;
    
    // Trier par F1-score décroissant (meilleur en premier)
    if (ra->avg_f1_score > rb->avg_f1_score) return -1;
    if (ra->avg_f1_score < rb->avg_f1_score) return 1;
    
    // En cas d'égalité, trier par accuracy
    if (ra->accuracy > rb->accuracy) return -1;
    if (ra->accuracy < rb->accuracy) return 1;
    
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <mnist_data_dir> <output_dir>\n", argv[0]);
        printf("Exemple: %s data/mnist models/\n", argv[0]);
        return 1;
    }
    
    const char *data_dir = argv[1];
    const char *output_dir = argv[2];
    
    srand(time(NULL));
    
    LOG_INFO("========================================");
    LOG_INFO("  GRID SEARCH - OPTIMISATION CNN");
    LOG_INFO("========================================\n");
    
    // Construire les chemins des fichiers MNIST
    char train_images[256], train_labels[256];
    char test_images[256], test_labels[256];
    
    snprintf(train_images, sizeof(train_images), "%s/train-images.idx3-ubyte", data_dir);
    snprintf(train_labels, sizeof(train_labels), "%s/train-labels.idx1-ubyte", data_dir);
    snprintf(test_images, sizeof(test_images), "%s/t10k-images.idx3-ubyte", data_dir);
    snprintf(test_labels, sizeof(test_labels), "%s/t10k-labels.idx1-ubyte", data_dir);
    
    // Charger les datasets
    LOG_INFO("Chargement des datasets...");
    MNISTDataset *train_data = load_mnist_dataset(train_images, train_labels);
    MNISTDataset *test_data = load_mnist_dataset(test_images, test_labels);
    
    if (!train_data || !test_data) {
        LOG_ERROR("Échec du chargement des datasets");
        return 1;
    }
    
    LOG_INFO("Dataset chargé: %zu train, %zu test\n", train_data->count, test_data->count);
    
    // Définir la grille de recherche (OPTIMISÉE - configurations prometteuses)
    // int epochs_grid[] = {10, 15, 20};
    // int batch_sizes[] = {32};
    // float learning_rates[] = {0.005f, 0.01f, 0.02f};
    // float momentums[] = {0.0f, 0.9f};

    int epochs_grid[] = {20};
    int batch_sizes[] = {32};
    float learning_rates[] = {0.005f, 0.01f, 0.02f};
    float momentums[] = {0.0f, 0.9f};
    
    int n_epochs = sizeof(epochs_grid) / sizeof(epochs_grid[0]);
    int n_batch = sizeof(batch_sizes) / sizeof(batch_sizes[0]);
    int n_lr = sizeof(learning_rates) / sizeof(learning_rates[0]);
    int n_momentum = sizeof(momentums) / sizeof(momentums[0]);
    
    int total_configs = n_epochs * n_batch * n_lr * n_momentum;
    
    LOG_INFO("Grille de recherche:");
    LOG_INFO("  - Epochs: %d valeurs", n_epochs);
    LOG_INFO("  - Batch sizes: %d valeurs", n_batch);
    LOG_INFO("  - Learning rates: %d valeurs", n_lr);
    LOG_INFO("  - Momentums: %d valeurs", n_momentum);
    LOG_INFO("  - Total configurations: %d\n", total_configs);
    
    // Allouer le tableau de résultats
    GridSearchResult *results = calloc(total_configs, sizeof(GridSearchResult));
    if (!results) {
        LOG_ERROR("Échec d'allocation mémoire");
        free_mnist_dataset(train_data);
        free_mnist_dataset(test_data);
        return 1;
    }
    
    // Lancer le grid search
    int config_idx = 0;
    clock_t total_start = clock();
    
    for (int e = 0; e < n_epochs; e++) {
        for (int b = 0; b < n_batch; b++) {
            for (int lr = 0; lr < n_lr; lr++) {
                for (int m = 0; m < n_momentum; m++) {
                    GridSearchResult *result = &results[config_idx];
                    
                    result->epochs = epochs_grid[e];
                    result->batch_size = batch_sizes[b];
                    result->learning_rate = learning_rates[lr];
                    result->momentum = momentums[m];
                    
                    LOG_INFO("========================================");
                    LOG_INFO("Configuration %d/%d", config_idx + 1, total_configs);
                    LOG_INFO("Epochs=%d, Batch=%d, LR=%.4f, Momentum=%.2f",
                             result->epochs, result->batch_size, 
                             result->learning_rate, result->momentum);
                    LOG_INFO("========================================");
                    
                    // Créer et entraîner le modèle
                    CNNModel *model = create_cnn_model();
                    if (!model) {
                        LOG_ERROR("Échec de création du modèle");
                        continue;
                    }
                    
                    clock_t start = clock();
                    
                    train_cnn(model, train_data, test_data,
                             result->epochs, result->batch_size, result->learning_rate);
                    
                    clock_t end = clock();
                    result->training_time = (double)(end - start) / CLOCKS_PER_SEC;
                    
                    // Calculer les métriques
                    compute_metrics(model, test_data, result);
                    
                    LOG_INFO("Résultats: Accuracy=%.2f%%, F1=%.4f, Time=%.2fmin",
                             result->accuracy * 100, result->avg_f1_score,
                             result->training_time / 60.0);
                    
                    free_cnn_model(model);
                    config_idx++;
                }
            }
        }
    }
    
    clock_t total_end = clock();
    double total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;
    
    LOG_INFO("\n========================================");
    LOG_INFO("GRID SEARCH TERMINÉ");
    LOG_INFO("========================================");
    LOG_INFO("Temps total: %.2f minutes", total_time / 60.0);
    LOG_INFO("Configurations testées: %d\n", config_idx);
    
    // Trier les résultats par F1-score
    qsort(results, config_idx, sizeof(GridSearchResult), compare_results);
    
    // Afficher le TOP 5
    LOG_INFO("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║                         TOP 5 CONFIGURATIONS                              ║");
    LOG_INFO("╚═══════════════════════════════════════════════════════════════════════════╝");
    
    printf("\n");
    printf("┌──────┬────────┬───────┬──────┬──────────┬──────────┬──────────┬──────────┐\n");
    printf("│ Rank │ Epochs │ Batch │  LR  │ Momentum │ Accuracy │   F1     │ Time(min)│\n");
    printf("├──────┼────────┼───────┼──────┼──────────┼──────────┼──────────┼──────────┤\n");
    
    int top_n = (config_idx < 5) ? config_idx : 5;
    for (int i = 0; i < top_n; i++) {
        GridSearchResult *r = &results[i];
        printf("│  %2d  │  %4d  │  %3d  │%.4f│   %.2f   │  %.4f  │  %.4f  │  %6.2f  │\n",
               i + 1, r->epochs, r->batch_size, r->learning_rate, r->momentum,
               r->accuracy, r->avg_f1_score, r->training_time / 60.0);
    }
    
    printf("└──────┴────────┴───────┴──────┴──────────┴──────────┴──────────┴──────────┘\n");
    
    // Afficher les détails de la meilleure configuration
    LOG_INFO("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║                    MEILLEURE CONFIGURATION                                ║");
    LOG_INFO("╚═══════════════════════════════════════════════════════════════════════════╝");
    
    print_metrics_summary(&results[0]);
    print_per_class_metrics(&results[0]);
    
    // Sauvegarder les résultats
    char csv_path[512];
    snprintf(csv_path, sizeof(csv_path), "%s/grid_search_results.csv", output_dir);
    save_results_to_csv(results, config_idx, csv_path);
    
    // Sauvegarder les meilleurs paramètres dans un fichier
    char best_params_path[512];
    snprintf(best_params_path, sizeof(best_params_path), "%s/best_params.txt", output_dir);
    FILE *f = fopen(best_params_path, "w");
    if (f) {
        fprintf(f, "# Meilleurs hyperparamètres trouvés par Grid Search\n");
        fprintf(f, "# Date: %s\n", __DATE__);
        fprintf(f, "\n");
        fprintf(f, "EPOCHS=%d\n", results[0].epochs);
        fprintf(f, "BATCH_SIZE=%d\n", results[0].batch_size);
        fprintf(f, "LEARNING_RATE=%.4f\n", results[0].learning_rate);
        fprintf(f, "MOMENTUM=%.2f\n", results[0].momentum);
        fprintf(f, "\n");
        fprintf(f, "# Métriques obtenues\n");
        fprintf(f, "ACCURACY=%.4f\n", results[0].accuracy);
        fprintf(f, "AVG_F1_SCORE=%.4f\n", results[0].avg_f1_score);
        fprintf(f, "TRAINING_TIME_MIN=%.2f\n", results[0].training_time / 60.0);
        fclose(f);
        LOG_INFO("\nMeilleurs paramètres sauvegardés: %s", best_params_path);
    }
    
    // Ré-entraîner avec la meilleure config et sauvegarder le modèle
    LOG_INFO("\n========================================");
    LOG_INFO("ENTRAÎNEMENT FINAL AVEC MEILLEURE CONFIG");
    LOG_INFO("========================================\n");
    
    CNNModel *final_model = create_cnn_model();
    if (final_model) {
        train_cnn(final_model, train_data, test_data,
                 results[0].epochs, results[0].batch_size, results[0].learning_rate);
        
        char weights_path[512];
        snprintf(weights_path, sizeof(weights_path), "%s/cnn_weights_optimized.bin", output_dir);
        if (save_cnn_weights(final_model, weights_path)) {
            LOG_INFO("Modèle optimisé sauvegardé: %s", weights_path);
        }
        
        free_cnn_model(final_model);
    }
    
    // Nettoyage
    free(results);
    free_mnist_dataset(train_data);
    free_mnist_dataset(test_data);
    
    LOG_INFO("\n========================================");
    LOG_INFO("Grid Search terminé avec succès!");
    LOG_INFO("========================================\n");
    
    return 0;
}
