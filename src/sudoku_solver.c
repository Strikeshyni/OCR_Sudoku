#include "sudoku_solver.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// CRÉATION ET LIBÉRATION
// ============================================================================

SudokuGrid* create_sudoku_grid() {
    SudokuGrid *grid = (SudokuGrid*)malloc(sizeof(SudokuGrid));
    memset(grid->grid, 0, sizeof(grid->grid));
    memset(grid->fixed, false, sizeof(grid->fixed));
    return grid;
}

void free_sudoku_grid(SudokuGrid *grid) {
    free(grid);
}

void fill_grid_from_digits(SudokuGrid *grid, const int digits[81]) {
    for (int i = 0; i < 81; i++) {
        int row = i / 9;
        int col = i % 9;
        grid->grid[row][col] = digits[i];
        grid->fixed[row][col] = (digits[i] != 0);
    }
}

// ============================================================================
// VALIDATION
// ============================================================================

bool is_valid_placement(const SudokuGrid *grid, int row, int col, int num) {
    // Vérifier la ligne
    for (int c = 0; c < 9; c++) {
        if (grid->grid[row][c] == num) return false;
    }
    
    // Vérifier la colonne
    for (int r = 0; r < 9; r++) {
        if (grid->grid[r][col] == num) return false;
    }
    
    // Vérifier le bloc 3x3
    int block_row = (row / 3) * 3;
    int block_col = (col / 3) * 3;
    
    for (int r = block_row; r < block_row + 3; r++) {
        for (int c = block_col; c < block_col + 3; c++) {
            if (grid->grid[r][c] == num) return false;
        }
    }
    
    return true;
}

bool is_grid_complete(const SudokuGrid *grid) {
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            if (grid->grid[row][col] == 0) return false;
            if (!is_valid_placement(grid, row, col, grid->grid[row][col])) {
                // Temporairement retirer le chiffre pour tester
                int temp = grid->grid[row][col];
                ((SudokuGrid*)grid)->grid[row][col] = 0;
                bool valid = is_valid_placement(grid, row, col, temp);
                ((SudokuGrid*)grid)->grid[row][col] = temp;
                if (!valid) return false;
            }
        }
    }
    return true;
}

// ============================================================================
// RÉSOLUTION BACKTRACKING SIMPLE
// ============================================================================

static bool solve_backtrack(SudokuGrid *grid, int pos) {
    if (pos == 81) {
        return true;  // Grille complète
    }
    
    int row = pos / 9;
    int col = pos % 9;
    
    // Si la case est déjà remplie (fixée), passer à la suivante
    if (grid->fixed[row][col]) {
        return solve_backtrack(grid, pos + 1);
    }
    
    // Essayer tous les chiffres de 1 à 9
    for (int num = 1; num <= 9; num++) {
        if (is_valid_placement(grid, row, col, num)) {
            grid->grid[row][col] = num;
            
            if (solve_backtrack(grid, pos + 1)) {
                return true;
            }
            
            // Backtrack
            grid->grid[row][col] = 0;
        }
    }
    
    return false;
}

bool solve_sudoku(SudokuGrid *grid) {
    LOG_INFO("Résolution de la grille Sudoku...");
    bool solved = solve_backtrack(grid, 0);
    
    if (solved) {
        LOG_INFO("Grille résolue avec succès!");
    } else {
        LOG_ERROR("Aucune solution trouvée");
    }
    
    return solved;
}

// ============================================================================
// RÉSOLUTION OPTIMISÉE (MRV - Minimum Remaining Values)
// ============================================================================

// Compte les valeurs possibles pour une case
static int count_possibilities(const SudokuGrid *grid, int row, int col) {
    if (grid->fixed[row][col]) return 0;
    
    int count = 0;
    for (int num = 1; num <= 9; num++) {
        if (is_valid_placement(grid, row, col, num)) {
            count++;
        }
    }
    return count;
}

// Trouve la case vide avec le moins de possibilités (MRV heuristic)
static bool find_best_cell(const SudokuGrid *grid, int *best_row, int *best_col) {
    int min_possibilities = 10;
    bool found = false;
    
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            if (grid->grid[row][col] == 0) {
                int poss = count_possibilities(grid, row, col);
                
                if (poss == 0) {
                    // Impasse détectée tôt
                    return false;
                }
                
                if (poss < min_possibilities) {
                    min_possibilities = poss;
                    *best_row = row;
                    *best_col = col;
                    found = true;
                    
                    // Optimisation: si seulement 1 possibilité, on prend immédiatement
                    if (poss == 1) return true;
                }
            }
        }
    }
    
    return found;
}

static bool solve_mrv_recursive(SudokuGrid *grid) {
    int row, col;
    
    // Trouver la meilleure case à remplir (MRV)
    if (!find_best_cell(grid, &row, &col)) {
        // Soit grille complète, soit impasse
        return is_grid_complete(grid);
    }
    
    // Essayer tous les chiffres valides
    for (int num = 1; num <= 9; num++) {
        if (is_valid_placement(grid, row, col, num)) {
            grid->grid[row][col] = num;
            
            if (solve_mrv_recursive(grid)) {
                return true;
            }
            
            // Backtrack
            grid->grid[row][col] = 0;
        }
    }
    
    return false;
}

bool solve_sudoku_mrv(SudokuGrid *grid) {
    LOG_INFO("Résolution de la grille Sudoku (MRV optimisé)...");
    bool solved = solve_mrv_recursive(grid);
    
    if (solved) {
        LOG_INFO("Grille résolue avec succès (MRV)!");
    } else {
        LOG_ERROR("Aucune solution trouvée (MRV)");
    }
    
    return solved;
}

// ============================================================================
// UNICITÉ DE SOLUTION
// ============================================================================

static int count_solutions_recursive(SudokuGrid *grid, int pos, int max_count, int *current_count) {
    if (*current_count >= max_count) return *current_count;
    
    if (pos == 81) {
        (*current_count)++;
        return *current_count;
    }
    
    int row = pos / 9;
    int col = pos % 9;
    
    if (grid->fixed[row][col] || grid->grid[row][col] != 0) {
        return count_solutions_recursive(grid, pos + 1, max_count, current_count);
    }
    
    for (int num = 1; num <= 9; num++) {
        if (is_valid_placement(grid, row, col, num)) {
            grid->grid[row][col] = num;
            count_solutions_recursive(grid, pos + 1, max_count, current_count);
            grid->grid[row][col] = 0;
            
            if (*current_count >= max_count) break;
        }
    }
    
    return *current_count;
}

int count_solutions(SudokuGrid *grid, int max_solutions) {
    int count = 0;
    count_solutions_recursive(grid, 0, max_solutions, &count);
    return count;
}

bool has_unique_solution(const SudokuGrid *grid) {
    SudokuGrid *test_grid = create_sudoku_grid();
    memcpy(test_grid, grid, sizeof(SudokuGrid));
    
    int num_solutions = count_solutions(test_grid, 2);
    
    free_sudoku_grid(test_grid);
    return num_solutions == 1;
}

// ============================================================================
// AFFICHAGE
// ============================================================================

void print_sudoku_grid(const SudokuGrid *grid) {
    printf("\n+---------+---------+---------+\n");
    for (int row = 0; row < 9; row++) {
        printf("|");
        for (int col = 0; col < 9; col++) {
            if (grid->grid[row][col] == 0) {
                printf(" . ");
            } else {
                printf(" %d ", grid->grid[row][col]);
            }
            
            if ((col + 1) % 3 == 0) {
                printf("|");
            }
        }
        printf("\n");
        
        if ((row + 1) % 3 == 0) {
            printf("+---------+---------+---------+\n");
        }
    }
}
