# Makefile pour OCR Sudoku Solver

CC = gcc
CFLAGS = -Wall -Wextra -O3 -std=c99 -march=native
LDFLAGS = -lm
DEBUG_FLAGS = -g -O0 -DDEBUG

SRC_DIR = src
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)

# Sources communes
COMMON_SRCS = $(SRC_DIR)/utils.c \
              $(SRC_DIR)/image_loader.c \
              $(SRC_DIR)/preprocessor.c \
              $(SRC_DIR)/grid_detector.c \
              $(SRC_DIR)/perspective.c \
              $(SRC_DIR)/cell_extractor.c \
              $(SRC_DIR)/cnn_model.c \
              $(SRC_DIR)/sudoku_solver.c \
              $(SRC_DIR)/image_composer.c

# Sources pour entraînement
TRAIN_SRCS = $(COMMON_SRCS) \
             $(SRC_DIR)/cnn_training.c \
             $(SRC_DIR)/dataset_loader.c \
             $(SRC_DIR)/train_cnn.c

# Sources pour grid search
GRID_SEARCH_SRCS = $(COMMON_SRCS) \
                   $(SRC_DIR)/cnn_training.c \
                   $(SRC_DIR)/dataset_loader.c \
                   $(SRC_DIR)/grid_search.c

# Sources pour exécution
MAIN_SRCS = $(COMMON_SRCS) \
            $(SRC_DIR)/main.c

# Objets
TRAIN_OBJS = $(TRAIN_SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
GRID_SEARCH_OBJS = $(GRID_SEARCH_SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
MAIN_OBJS = $(MAIN_SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

# Exécutables
TRAIN_BIN = $(BIN_DIR)/train_cnn
GRID_SEARCH_BIN = $(BIN_DIR)/grid_search
MAIN_BIN = $(BIN_DIR)/sudoku_solver

.PHONY: all clean train gridsearch run debug directories

all: directories $(MAIN_BIN)

train: directories $(TRAIN_BIN)
	@echo "Entraînement du CNN..."
	$(TRAIN_BIN) data/mnist models/cnn_weights.bin

gridsearch: directories $(GRID_SEARCH_BIN)
	@echo "Lancement du Grid Search..."
	$(GRID_SEARCH_BIN) data/mnist models/

directories:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR) models data/mnist data/test_images

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(MAIN_BIN): $(MAIN_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✓ Compilation réussie: $@"

$(TRAIN_BIN): $(TRAIN_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✓ Compilation réussie: $@"

$(GRID_SEARCH_BIN): $(GRID_SEARCH_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✓ Compilation réussie: $@"

debug: CFLAGS = -Wall -Wextra $(DEBUG_FLAGS) -std=c99
debug: clean all

run: $(MAIN_BIN)
	@echo "Usage: $(MAIN_BIN) <input_image.jpg> <output_image.png>"

clean:
	rm -rf $(BUILD_DIR)
	@echo "✓ Nettoyage terminé"

install:
	@echo "Installation des headers stb..."
	@mkdir -p $(SRC_DIR)/external
	@if [ ! -f $(SRC_DIR)/external/stb_image.h ]; then \
		wget -q https://raw.githubusercontent.com/nothings/stb/master/stb_image.h -O $(SRC_DIR)/external/stb_image.h; \
		echo "✓ stb_image.h téléchargé"; \
	fi
	@if [ ! -f $(SRC_DIR)/external/stb_image_write.h ]; then \
		wget -q https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h -O $(SRC_DIR)/external/stb_image_write.h; \
		echo "✓ stb_image_write.h téléchargé"; \
	fi

help:
	@echo "Commandes disponibles:"
	@echo "  make all        - Compiler le solveur Sudoku"
	@echo "  make train      - Compiler et entraîner le CNN"
	@echo "  make gridsearch - Lancer le Grid Search pour optimiser les hyperparamètres"
	@echo "  make debug      - Compiler en mode debug"
	@echo "  make clean      - Nettoyer les fichiers compilés"
	@echo "  make install    - Télécharger les dépendances (stb)"
	@echo "  make help       - Afficher cette aide"
