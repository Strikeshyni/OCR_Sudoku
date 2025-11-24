cmake_minimum_required(VERSION 3.10)
project(OCR_Sudoku C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3 -march=native")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0 -DDEBUG")

# Sources communes
set(COMMON_SOURCES
    src/utils.c
    src/image_loader.c
    src/preprocessor.c
    src/grid_detector.c
    src/perspective.c
    src/cell_extractor.c
    src/cnn_model.c
    src/sudoku_solver.c
    src/image_composer.c
)

# Exécutable principal
add_executable(sudoku_solver
    ${COMMON_SOURCES}
    src/main.c
)

# Exécutable d'entraînement
add_executable(train_cnn
    ${COMMON_SOURCES}
    src/cnn_training.c
    src/dataset_loader.c
    src/train_cnn.c
)

# Librairie mathématique
target_link_libraries(sudoku_solver m)
target_link_libraries(train_cnn m)

# Création des dossiers
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/models)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data/mnist)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data/test_images)
