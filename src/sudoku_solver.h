#ifndef SUDOKU_SOLVER_H
#define SUDOKU_SOLVER_H

#include <stdbool.h>
#include <stdint.h>

// ============================================================================
// STRUCTURES
// ============================================================================

// Grille de Sudoku (9x9)
typedef struct {
    int grid[9][9];         // 0 = case vide, 1-9 = chiffres
    bool fixed[9][9];       // true si la case était dans la grille originale
} SudokuGrid;

// ============================================================================
// RÉSOLUTION
// ============================================================================

// Résout une grille de Sudoku par backtracking optimisé
// grid: grille à résoudre (modifiée en place)
// Returns: true si solution trouvée, false sinon
bool solve_sudoku(SudokuGrid *grid);

// Résout avec heuristique MRV (Minimum Remaining Values)
bool solve_sudoku_mrv(SudokuGrid *grid);

// ============================================================================
// VALIDATION
// ============================================================================

// Vérifie si un nombre peut être placé à une position donnée
bool is_valid_placement(const SudokuGrid *grid, int row, int col, int num);

// Vérifie si la grille est complète et valide
bool is_grid_complete(const SudokuGrid *grid);

// Vérifie si la grille a une solution unique
bool has_unique_solution(const SudokuGrid *grid);

// ============================================================================
// UTILITAIRES
// ============================================================================

// Crée une grille vide
SudokuGrid* create_sudoku_grid();

// Libère une grille
void free_sudoku_grid(SudokuGrid *grid);

// Remplit une grille depuis un tableau de chiffres reconnus
void fill_grid_from_digits(SudokuGrid *grid, const int digits[81]);

// Affiche une grille (debug)
void print_sudoku_grid(const SudokuGrid *grid);

// Compte le nombre de solutions possibles (limité à max_solutions)
int count_solutions(SudokuGrid *grid, int max_solutions);

#endif // SUDOKU_SOLVER_H
