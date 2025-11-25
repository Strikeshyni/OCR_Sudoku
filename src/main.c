#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "image_loader.h"
#include "preprocessor.h"
#include "grid_detector.h"
#include "perspective.h"
#include "cell_extractor.h"
#include "cnn_model.h"
#include "sudoku_solver.h"
#include "image_composer.h"

// ============================================================================
// PREDICTION CORRECTION & BACKTRACKING
// ============================================================================

// Global counter for backtracking steps
static int backtrack_count = 0;
#define MAX_BACKTRACKS 100000 // Limit to prevent infinite loops

typedef struct {
    int digit;
    float prob;
} Candidate;

typedef struct {
    Candidate candidates[10];
    int count;
} CellCandidates;

int compare_candidates(const void *a, const void *b) {
    float diff = ((Candidate*)b)->prob - ((Candidate*)a)->prob;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}

// Simple recursive flood fill for cleaning cell borders
// Keeps only the component connected to (start_x, start_y)
// Returns area
int flood_fill_keep(uint8_t *data, int w, int h, int x, int y, uint8_t *mask) {
    if (x < 0 || x >= w || y < 0 || y >= h) return 0;
    if (mask[y * w + x] == 1) return 0; // Already visited
    if (data[y * w + x] < 128) return 0; // Background (black)
    
    mask[y * w + x] = 1; // Mark as kept
    int area = 1;
    
    area += flood_fill_keep(data, w, h, x + 1, y, mask);
    area += flood_fill_keep(data, w, h, x - 1, y, mask);
    area += flood_fill_keep(data, w, h, x, y + 1, mask);
    area += flood_fill_keep(data, w, h, x, y - 1, mask);
    
    return area;
}

void remove_border_noise(GrayImage *cell) {
    // Strategy: Find the largest connected component closest to the center.
    // Keep it, remove everything else.
    // Assumes cell is inverted (white digit on black background).
    
    int w = cell->width;
    int h = cell->height;
    uint8_t *mask = (uint8_t*)calloc(w * h, sizeof(uint8_t));
    
    int max_area = 0;
    int best_seed_x = -1;
    int best_seed_y = -1;
    
    // Find all components
    uint8_t *visited = (uint8_t*)calloc(w * h, sizeof(uint8_t));
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (cell->data[y * w + x] > 128 && visited[y * w + x] == 0) {
                // Found a new component
                uint8_t *temp_mask = (uint8_t*)calloc(w * h, sizeof(uint8_t));
                int area = flood_fill_keep(cell->data, w, h, x, y, temp_mask);
                
                // Mark as visited globally
                for(int i=0; i<w*h; i++) {
                    if(temp_mask[i]) visited[i] = 1;
                }
                
                // Check if this component is better (larger and/or more central)
                // For digits, we want the largest component usually.
                // But sometimes noise is large? Unlikely for border lines if cropped well.
                // Let's just take the largest component.
                if (area > max_area) {
                    max_area = area;
                    best_seed_x = x;
                    best_seed_y = y;
                }
                
                free(temp_mask);
            }
        }
    }
    
    free(visited);
    
    // If we found a component, keep only that one
    if (best_seed_x != -1) {
        // Re-run flood fill to get the mask of the best component
        memset(mask, 0, w * h);
        flood_fill_keep(cell->data, w, h, best_seed_x, best_seed_y, mask);
        
        // Apply mask: set everything else to 0
        for (int i = 0; i < w * h; i++) {
            if (mask[i] == 0) {
                cell->data[i] = 0;
            }
        }
    } else {
        // No components found (empty cell?), clear everything just in case
        memset(cell->data, 0, w * h);
    }
    
    free(mask);
}

bool is_safe_partial(int *grid, int index, int digit) {
    int row = index / 9;
    int col = index % 9;
    
    // Check row
    for (int c = 0; c < 9; c++) {
        int idx = row * 9 + c;
        if (idx != index && grid[idx] == digit) return false;
    }
    
    // Check col
    for (int r = 0; r < 9; r++) {
        int idx = r * 9 + col;
        if (idx != index && grid[idx] == digit) return false;
    }
    
    // Check box
    int start_r = (row / 3) * 3;
    int start_c = (col / 3) * 3;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            int idx = (start_r + r) * 9 + (start_c + c);
            if (idx != index && grid[idx] == digit) return false;
        }
    }
    
    return true;
}

// Structure to sort cells by confidence
typedef struct {
    int index;
    float max_prob;
} CellConfidence;

int compare_cell_confidence(const void *a, const void *b) {
    float diff = ((CellConfidence*)b)->max_prob - ((CellConfidence*)a)->max_prob;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}

bool find_valid_clues(int step, int *processing_order, CellCandidates *candidates, int *current_grid, SudokuGrid *result) {
    // Check timeout
    backtrack_count++;
    if (backtrack_count > MAX_BACKTRACKS) {
        return false; // Timeout
    }

    if (step == 81) {
        // All clues placed and consistent. Now check if solvable.
        SudokuGrid s_grid;
        for(int i=0; i<81; i++) {
            s_grid.grid[i/9][i%9] = current_grid[i];
            s_grid.fixed[i/9][i%9] = (current_grid[i] != 0);
        }
        
        // We need a copy because solve_sudoku modifies it
        SudokuGrid to_solve = s_grid;
        if (solve_sudoku(&to_solve)) {
            *result = to_solve;
            return true;
        }
        return false;
    }
    
    int index = processing_order[step];
    
    if (candidates[index].count == 0) {
        current_grid[index] = 0;
        return find_valid_clues(step + 1, processing_order, candidates, current_grid, result);
    }
    
    // Try candidates (sorted by probability)
    // We limit to top 5 to ensure performance, though usually top 2-3 is enough
    int max_tries = candidates[index].count;
    if (max_tries > 5) max_tries = 5;

    for (int i = 0; i < max_tries; i++) {
        int digit = candidates[index].candidates[i].digit;
        
        current_grid[index] = digit;
        if (is_safe_partial(current_grid, index, digit)) {
            if (find_valid_clues(step + 1, processing_order, candidates, current_grid, result)) return true;
        }
    }
    
    current_grid[index] = 0; // Backtrack
    return false;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];

    printf("Loading image: %s\n", input_path);
    RGBImage *original = load_rgb_image(input_path);
    if (!original) {
        fprintf(stderr, "Failed to load image\n");
        return 1;
    }

    // 1. Preprocessing
    printf("Preprocessing...\n");
    GrayImage *gray = rgb_to_gray(original);
    save_gray_image("debug_1_gray.png", gray);

    GrayImage *blurred = gaussian_blur(gray, 5, 1.0f);
    save_gray_image("debug_2_blurred.png", blurred);
    
    // Adaptive threshold for grid detection
    GrayImage *binary = gray_image_clone(blurred);
    threshold_otsu(binary);
    invert_image(binary); // Make lines white on black
    
    // Create a dilated copy for grid detection (connects broken lines)
    GrayImage *binary_dilated = gray_image_clone(binary);
    dilate(binary_dilated, 3);
    save_gray_image("debug_3_binary.png", binary_dilated);

    // 2. Grid Detection
    printf("Detecting grid...\n");
    Quad grid_quad;
    // Use binary_dilated for detection as it connects lines better
    if (!find_largest_quad(binary_dilated, &grid_quad)) {
        fprintf(stderr, "Failed to detect grid\n");
        // Cleanup
        return 1;
    }
    printf("Grid detected!\n");
    
    // Free dilated image, we don't need it anymore
    free(binary_dilated->data);
    free(binary_dilated);

    // Debug: Draw detected grid on BINARY image (converted to RGB)
    // We want to see the grid on the image we actually used (or close to it)
    // But user asked to use binary for everything.
    // Let's visualize on the binary image to be consistent.
    RGBImage *debug_grid = rgb_image_create(binary->width, binary->height, 3);
    for(int i=0; i<binary->width*binary->height; i++) {
        uint8_t val = binary->data[i];
        debug_grid->data[i*3] = val;
        debug_grid->data[i*3+1] = val;
        debug_grid->data[i*3+2] = val;
    }
    
    for (int i = 0; i < 4; i++) {
        draw_line_rgb(debug_grid, grid_quad.corners[i], grid_quad.corners[(i+1)%4], 0, 255, 0, 3);
    }
    save_rgb_image("debug_4_grid_detected.png", debug_grid);
    rgb_image_free(debug_grid);

    // 3. Perspective Transform
    printf("Rectifying grid...\n");
    
    // Define destination quad (square)
    Quad dst_quad;
    int size = 252; // 28 * 9
    dst_quad.corners[0] = (Point2D){0, 0};
    dst_quad.corners[1] = (Point2D){size, 0};
    dst_quad.corners[2] = (Point2D){size, size};
    dst_quad.corners[3] = (Point2D){0, size};
    
    HomographyMatrix H = compute_homography(&grid_quad, &dst_quad);
    // Use the binary image for warping to get clean, thresholded cells
    GrayImage *rectified = warp_perspective(binary, &H, size, size);
    save_gray_image("debug_5_rectified.png", rectified);
    
    // 4. Cell Extraction
    printf("Extracting cells...\n");
    GrayImage *cells[81];
    if (!extract_sudoku_cells(rectified, cells)) {
        fprintf(stderr, "Failed to extract cells\n");
        return 1;
    }

    // Invert cells for CNN (MNIST expects white digits on black background)
    // and create debug image with red borders
    printf("Inverting cells and creating debug image...\n");
    
    // Create RGB image for debug to support colored borders
    // Grid size: 9x9 cells. Each cell 28x28.
    // Let's add 1px border between cells.
    // Total width = 9 * 28 + 10 * 1 = 252 + 10 = 262
    int border = 1;
    int cell_size = 28;
    int grid_img_size = 9 * cell_size + 10 * border;
    
    RGBImage *cells_grid = rgb_image_create(grid_img_size, grid_img_size, 3);
    // Fill with red (borders)
    for(int i=0; i<grid_img_size * grid_img_size * 3; i+=3) {
        cells_grid->data[i] = 255;   // R
        cells_grid->data[i+1] = 0;   // G
        cells_grid->data[i+2] = 0;   // B
    }
    
    for (int i = 0; i < 81; i++) {
        int r = i / 9;
        int c = i % 9;
        
        if (cells[i]) {
            // Cells are already inverted (white on black) because we warped the inverted binary image.
            // So we DO NOT invert them again.
            
            // Remove border noise (keep only largest component)
            remove_border_noise(cells[i]);
            
            // Copy to debug image
            int start_y = border + r * (cell_size + border);
            int start_x = border + c * (cell_size + border);
            
            for (int y = 0; y < cell_size; y++) {
                for (int x = 0; x < cell_size; x++) {
                    int dest_idx = ((start_y + y) * grid_img_size + (start_x + x)) * 3;
                    uint8_t val = cells[i]->data[y * cell_size + x];
                    cells_grid->data[dest_idx] = val;
                    cells_grid->data[dest_idx+1] = val;
                    cells_grid->data[dest_idx+2] = val;
                }
            }
        }
    }
    save_rgb_image("debug_6_cells.png", cells_grid);
    rgb_image_free(cells_grid);

    // 5. CNN Recognition
    printf("Recognizing digits...\n");
    CNNModel *model = create_cnn_model();
    if (!load_cnn_weights(model, "models/cnn_weights.bin")) {
        fprintf(stderr, "Failed to load CNN weights\n");
        // Try default path or warn
        printf("Warning: Using random weights (for testing only)\n");
    }

    // Prepare candidates for backtracking
    CellCandidates cell_candidates[81];
    CellConfidence cell_confidences[81];
    
    printf("\n=== Raw Predictions ===\n");
    printf("Row | Col | Empty? | Top 1 (Prob)| Top 2 (Prob)| Top 3 (Prob)\n");
    printf("----|-----|--------|-------------|-------------|-------------\n");

    for (int i = 0; i < 81; i++) {
        cell_candidates[i].count = 0;
        cell_confidences[i].index = i;
        cell_confidences[i].max_prob = 0.0f;
        
        int r = i / 9;
        int c = i % 9;
        
        if (is_cell_empty(cells[i])) {
            // Empty cell
            cell_candidates[i].count = 0; 
            printf("  %d |  %d  |  YES   |      -      |      -      |      -\n", r, c);
        } else {
            // Get probabilities
            float *input = prepare_cell_for_cnn(cells[i]);
            float *probs = cnn_forward(model, input);
            free(input);
            
            // Store candidates
            for(int d=1; d<=9; d++) { // Only 1-9 are valid for Sudoku
                 cell_candidates[i].candidates[cell_candidates[i].count].digit = d;
                 cell_candidates[i].candidates[cell_candidates[i].count].prob = probs[d];
                 cell_candidates[i].count++;
            }
            
            // Find max prob for sorting
            float max_p = 0;
            for(int d=1; d<=9; d++) {
                if(probs[d] > max_p) max_p = probs[d];
            }
            cell_confidences[i].max_prob = max_p;
            
            free(probs);
            
            // Sort candidates by probability (descending)
            qsort(cell_candidates[i].candidates, cell_candidates[i].count, sizeof(Candidate), compare_candidates);
            
            // Print top 3 predictions
            printf("  %d |  %d  |   NO   |  %d (%5.1f%%) |  %d (%5.1f%%) |  %d (%5.1f%%)\n", 
                   r, c,
                   cell_candidates[i].candidates[0].digit, cell_candidates[i].candidates[0].prob * 100,
                   cell_candidates[i].candidates[1].digit, cell_candidates[i].candidates[1].prob * 100,
                   cell_candidates[i].candidates[2].digit, cell_candidates[i].candidates[2].prob * 100);
        }
        
        // Free cell image
        free(cells[i]->data);
        free(cells[i]);
    }
    printf("=======================\n\n");
    
    // Sort cells by confidence (process most confident first)
    qsort(cell_confidences, 81, sizeof(CellConfidence), compare_cell_confidence);
    
    int processing_order[81];
    for(int i=0; i<81; i++) {
        processing_order[i] = cell_confidences[i].index;
    }
    
    // Solve with backtracking on clues
    int current_grid[81];
    // Initialize current_grid to 0
    memset(current_grid, 0, 81 * sizeof(int));
    
    SudokuGrid s_grid;
    
    // Reset backtrack counter
    backtrack_count = 0;
    
    printf("Searching for valid grid configuration using probabilistic backtracking (sorted by confidence)...\n");
    if (find_valid_clues(0, processing_order, cell_candidates, current_grid, &s_grid)) {
         printf("Valid grid found and solved!\n");
    } else {
         fprintf(stderr, "Could not find a valid grid configuration.\n");
         return 1;
    }
    
    // Print detected Grid (Initial clues that worked)
    printf("Detected Grid (Corrected):\n");
    for (int r = 0; r < 9; r++) {
        if (r % 3 == 0) printf("+-------+-------+-------+\n");
        for (int c = 0; c < 9; c++) {
            if (c % 3 == 0) printf("| ");
            // We can reconstruct the initial grid from s_grid.fixed
            // But s_grid contains the solution.
            // The fixed cells are the clues.
            if (s_grid.fixed[r][c])
                printf("%d ", s_grid.grid[r][c]);
            else
                printf(". ");
        }
        printf("|\n");
    }
    printf("+-------+-------+-------+\n");

    // 6. Solve Sudoku (Already done in find_valid_clues)
    // Just print the solution
    printf("Sudoku Solved!\n");

    // 7. Reconstruct Image
    printf("Composing output...\n");
    
    // Reconstruct initial_s_grid for display
    SudokuGrid initial_s_grid;
    for(int r=0; r<9; r++) {
        for(int c=0; c<9; c++) {
            if (s_grid.fixed[r][c]) {
                initial_s_grid.grid[r][c] = s_grid.grid[r][c];
                initial_s_grid.fixed[r][c] = true;
            } else {
                initial_s_grid.grid[r][c] = 0;
                initial_s_grid.fixed[r][c] = false;
            }
        }
    }
    
    RGBImage *output = compose_solved_image(gray, &initial_s_grid, &s_grid, &grid_quad);
    if (output) {
        save_rgb_image(output_path, output);
        // free output (need RGBImage free function)
        free(output->data);
        free(output);
    } else {
        printf("Could not compose output image.\n");
    }

    printf("Done. Saved to %s\n", output_path);

    // Cleanup
    free_cnn_model(model);
    // free images...
    
    return 0;
}

