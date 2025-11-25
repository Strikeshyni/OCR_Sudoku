#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"
#include "cnn_model.h"
#include "cnn_training.h"
#include "dataset_loader.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <mnist_data_dir> <output_weights_file>\n", argv[0]);
        printf("Exemple: %s data/mnist models/cnn_weights.bin\n", argv[0]);
        return 1;
    }
    
    const char *data_dir = argv[1];
    const char *output_file = argv[2];
    
    // Initialiser le générateur aléatoire
    srand(time(NULL));
    
    LOG_INFO("========================================");
    LOG_INFO("  ENTRAÎNEMENT CNN POUR RECONNAISSANCE");
    LOG_INFO("  DE CHIFFRES (MNIST)");
    LOG_INFO("========================================\n");
    
    // Construire les chemins des fichiers MNIST
    char train_images[256], train_labels[256];
    char test_images[256], test_labels[256];
    
    snprintf(train_images, sizeof(train_images), "%s/train-images.idx3-ubyte", data_dir);
    snprintf(train_labels, sizeof(train_labels), "%s/train-labels.idx1-ubyte", data_dir);
    snprintf(test_images, sizeof(test_images), "%s/t10k-images.idx3-ubyte", data_dir);
    snprintf(test_labels, sizeof(test_labels), "%s/t10k-labels.idx1-ubyte", data_dir);
    
    // Charger les datasets
    LOG_INFO("Chargement du dataset d'entraînement...");
    MNISTDataset *train_data = load_mnist_dataset(train_images, train_labels);
    if (!train_data) {
        LOG_ERROR("Échec du chargement du dataset d'entraînement");
        return 1;
    }
    
    LOG_INFO("Chargement du dataset de test...");
    MNISTDataset *test_data = load_mnist_dataset(test_images, test_labels);
    if (!test_data) {
        LOG_ERROR("Échec du chargement du dataset de test");
        free_mnist_dataset(train_data);
        return 1;
    }
    
    // Charger les données supplémentaires (Digital Digits)
    LOG_INFO("Recherche de données supplémentaires...");
    load_extra_dataset("data/digital_train.bin", train_data);
    load_extra_dataset("data/digital_test.bin", test_data);
    
    // Générer des échantillons vides (classe 0) pour remplacer les 0 filtrés
    // On vise environ 10% du dataset total pour que le modèle apprenne bien la classe "vide"
    LOG_INFO("Génération de la classe 'Vide' (0)...");
    int train_empty_count = train_data->count / 9; 
    int test_empty_count = test_data->count / 9;
    
    generate_empty_samples(train_data, train_empty_count);
    generate_empty_samples(test_data, test_empty_count);
    
    LOG_INFO("\nDataset chargé (avec classe Vide générée):");
    LOG_INFO("  - Entraînement: %zu images", train_data->count);
    LOG_INFO("  - Test: %zu images\n", test_data->count);
    
    // Créer le modèle CNN
    LOG_INFO("Création du modèle CNN...");
    CNNModel *model = create_cnn_model();
    if (!model) {
        LOG_ERROR("Échec de la création du modèle");
        free_mnist_dataset(train_data);
        free_mnist_dataset(test_data);
        return 1;
    }
    
    // Paramètres d'entraînement (par défaut ou depuis best_params.txt)
    int epochs = 50;
    int batch_size = 32;
    float learning_rate = 0.01f;
    
    // Essayer de charger les meilleurs paramètres si disponibles
    FILE *params_file = fopen("models/best_params.txt", "r");
    if (params_file) {
        LOG_INFO("Chargement des meilleurs paramètres depuis models/best_params.txt...");
        char line[256];
        while (fgets(line, sizeof(line), params_file)) {
            if (line[0] == '#' || line[0] == '\n') continue;  // Ignorer commentaires
            
            if (sscanf(line, "EPOCHS=%d", &epochs) == 1) continue;
            if (sscanf(line, "BATCH_SIZE=%d", &batch_size) == 1) continue;
            if (sscanf(line, "LEARNING_RATE=%f", &learning_rate) == 1) continue;
        }
        fclose(params_file);
        LOG_INFO("✓ Paramètres optimisés chargés\n");
    } else {
        LOG_INFO("Utilisation des paramètres par défaut (pas de best_params.txt)\n");
    }
    
    LOG_INFO("Paramètres d'entraînement:");
    LOG_INFO("  - Époques: %d", epochs);
    LOG_INFO("  - Batch size: %d", batch_size);
    LOG_INFO("  - Learning rate: %.4f\n", learning_rate);
    
    // Entraîner le modèle
    LOG_INFO("Début de l'entraînement...\n");
    clock_t start = clock();
    
    float final_accuracy = train_cnn(model, train_data, test_data, 
                                     epochs, batch_size, learning_rate);
    
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    LOG_INFO("\n========================================");
    LOG_INFO("ENTRAÎNEMENT TERMINÉ");
    LOG_INFO("========================================");
    LOG_INFO("Temps total: %.2f secondes (%.2f minutes)", elapsed, elapsed / 60.0);
    LOG_INFO("Précision finale: %.2f%%", final_accuracy * 100);
    
    // Sauvegarder le modèle final
    LOG_INFO("\nSauvegarde du modèle final...");
    if (save_cnn_weights(model, output_file)) {
        LOG_INFO("Modèle sauvegardé: %s", output_file);
    } else {
        LOG_ERROR("Échec de la sauvegarde du modèle");
    }
    
    // Évaluation finale sur le dataset de test
    LOG_INFO("\nÉvaluation finale sur le dataset de test...");
    float test_accuracy = evaluate_cnn(model, test_data);
    LOG_INFO("Précision sur test: %.2f%%", test_accuracy * 100);
    
    // Tests sur quelques exemples
    LOG_INFO("\nTests sur 10 exemples aléatoires:");
    for (int i = 0; i < 10; i++) {
        int idx = rand() % test_data->count;
        int predicted = cnn_predict(model, test_data->images[idx]);
        int actual = test_data->labels[idx];
        
        char status = (predicted == actual) ? '✓' : '✗';
        printf("  [%c] Exemple %d: Prédit=%d, Réel=%d\n", 
               status, idx, predicted, actual);
    }
    
    // Nettoyage
    free_cnn_model(model);
    free_mnist_dataset(train_data);
    free_mnist_dataset(test_data);
    
    LOG_INFO("\n========================================");
    LOG_INFO("Entraînement terminé avec succès!");
    LOG_INFO("Utilisez le fichier '%s' avec le solveur Sudoku", output_file);
    LOG_INFO("========================================\n");
    
    return 0;
}
