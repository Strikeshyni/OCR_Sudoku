#include "grid_detector.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG2RAD(x) ((x) * M_PI / 180.0f)
#define RAD2DEG(x) ((x) * 180.0f / M_PI)

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static int compare_lines(const void *a, const void *b) {
    HoughLine *la = (HoughLine *)a;
    HoughLine *lb = (HoughLine *)b;
    return lb->votes - la->votes; // Descending order
}

static Point2D intersection(HoughLine l1, HoughLine l2) {
    Point2D pt = {0, 0};
    float theta1 = l1.theta;
    float theta2 = l2.theta;
    float rho1 = l1.rho;
    float rho2 = l2.rho;

    float det = cos(theta1) * sin(theta2) - sin(theta1) * cos(theta2);
    if (fabs(det) < 1e-5) return pt; // Parallel lines

    pt.x = (sin(theta2) * rho1 - sin(theta1) * rho2) / det;
    pt.y = (cos(theta1) * rho2 - cos(theta2) * rho1) / det;
    return pt;
}

// ============================================================================
// HOUGH TRANSFORM
// ============================================================================

HoughLine* hough_lines(const GrayImage *edges, int threshold, int *num_lines) {
    int width = edges->width;
    int height = edges->height;
    int diagonal = (int)sqrt(width * width + height * height);
    int rho_len = 2 * diagonal + 1;
    int theta_len = 180; // 1 degree resolution

    // Allocate accumulator
    int *accumulator = (int *)calloc(rho_len * theta_len, sizeof(int));
    if (!accumulator) return NULL;

    // Precompute sin/cos
    float *sin_table = (float *)malloc(theta_len * sizeof(float));
    float *cos_table = (float *)malloc(theta_len * sizeof(float));
    for (int t = 0; t < theta_len; t++) {
        float theta = DEG2RAD(t);
        sin_table[t] = sin(theta);
        cos_table[t] = cos(theta);
    }

    // Fill accumulator
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (edges->data[y * width + x] > 128) { // Edge pixel
                for (int t = 0; t < theta_len; t++) {
                    int rho = (int)(x * cos_table[t] + y * sin_table[t]);
                    int rho_idx = rho + diagonal;
                    if (rho_idx >= 0 && rho_idx < rho_len) {
                        accumulator[rho_idx * theta_len + t]++;
                    }
                }
            }
        }
    }

    // Find lines above threshold
    int capacity = 1000;
    HoughLine *lines = (HoughLine *)malloc(capacity * sizeof(HoughLine));
    int count = 0;

    for (int r = 0; r < rho_len; r++) {
        for (int t = 0; t < theta_len; t++) {
            int votes = accumulator[r * theta_len + t];
            if (votes > threshold) {
                // Local maximum check (simple 3x3 window)
                bool is_max = true;
                for (int dr = -1; dr <= 1; dr++) {
                    for (int dt = -1; dt <= 1; dt++) {
                        if (dr == 0 && dt == 0) continue;
                        int nr = r + dr;
                        int nt = t + dt;
                        if (nr >= 0 && nr < rho_len && nt >= 0 && nt < theta_len) {
                            if (accumulator[nr * theta_len + nt] > votes) {
                                is_max = false;
                                break;
                            }
                        }
                    }
                    if (!is_max) break;
                }

                if (is_max) {
                    if (count >= capacity) {
                        capacity *= 2;
                        lines = (HoughLine *)realloc(lines, capacity * sizeof(HoughLine));
                    }
                    lines[count].rho = (float)(r - diagonal);
                    lines[count].theta = DEG2RAD(t);
                    lines[count].votes = votes;
                    count++;
                }
            }
        }
    }

    free(accumulator);
    free(sin_table);
    free(cos_table);

    // Sort lines by votes
    qsort(lines, count, sizeof(HoughLine), compare_lines);

    *num_lines = count;
    return lines;
}

// ============================================================================
// BLOB DETECTION (CONNECTED COMPONENTS)
// ============================================================================

// Simple stack-based flood fill to find connected components
// Returns the area of the component
static int flood_fill(int *labels, int width, int height, int x, int y, int label, 
                      int *min_x, int *max_x, int *min_y, int *max_y) {
    if (x < 0 || x >= width || y < 0 || y >= height) return 0;
    if (labels[y * width + x] != -1) return 0; // Already visited or background
    
    // Stack for recursion simulation
    int capacity = 10000;
    int *stack_x = (int*)malloc(capacity * sizeof(int));
    int *stack_y = (int*)malloc(capacity * sizeof(int));
    int top = 0;
    
    stack_x[top] = x;
    stack_y[top] = y;
    top++;
    
    int area = 0;
    *min_x = width; *max_x = 0;
    *min_y = height; *max_y = 0;
    
    while (top > 0) {
        top--;
        int cx = stack_x[top];
        int cy = stack_y[top];
        
        if (cx < 0 || cx >= width || cy < 0 || cy >= height) continue;
        if (labels[cy * width + cx] != -1) continue;
        
        labels[cy * width + cx] = label;
        area++;
        
        if (cx < *min_x) *min_x = cx;
        if (cx > *max_x) *max_x = cx;
        if (cy < *min_y) *min_y = cy;
        if (cy > *max_y) *max_y = cy;
        
        // Check neighbors (4-connectivity)
        if (top + 4 >= capacity) {
            capacity *= 2;
            stack_x = (int*)realloc(stack_x, capacity * sizeof(int));
            stack_y = (int*)realloc(stack_y, capacity * sizeof(int));
        }
        
        // Push neighbors if they are foreground (-1 means foreground not visited, 0 means background)
        // Wait, we need to know which pixels are foreground.
        // The labels array should be initialized: 0 for background, -1 for foreground.
        
        // Right
        if (cx + 1 < width && labels[cy * width + (cx + 1)] == -1) {
            stack_x[top] = cx + 1; stack_y[top] = cy; top++;
        }
        // Left
        if (cx - 1 >= 0 && labels[cy * width + (cx - 1)] == -1) {
            stack_x[top] = cx - 1; stack_y[top] = cy; top++;
        }
        // Down
        if (cy + 1 < height && labels[(cy + 1) * width + cx] == -1) {
            stack_x[top] = cx; stack_y[top] = cy + 1; top++;
        }
        // Up
        if (cy - 1 >= 0 && labels[(cy - 1) * width + cx] == -1) {
            stack_x[top] = cx; stack_y[top] = cy - 1; top++;
        }
    }
    
    free(stack_x);
    free(stack_y);
    return area;
}

// Find the largest connected component in the binary image
// Returns a mask where the largest component is 255, others 0
// Also returns the bounding box of the component
static GrayImage* find_largest_blob(const GrayImage *binary, int *bbox_x, int *bbox_y, int *bbox_w, int *bbox_h) {
    int w = binary->width;
    int h = binary->height;
    
    // Initialize labels: 0 for background, -1 for foreground
    int *labels = (int*)malloc(w * h * sizeof(int));
    for (int i = 0; i < w * h; i++) {
        labels[i] = (binary->data[i] > 128) ? -1 : 0;
    }
    
    int max_area = 0;
    int best_label = 0;
    int current_label = 1;
    
    int best_min_x = 0, best_max_x = 0, best_min_y = 0, best_max_y = 0;
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (labels[y * w + x] == -1) {
                int min_x, max_x, min_y, max_y;
                int area = flood_fill(labels, w, h, x, y, current_label, &min_x, &max_x, &min_y, &max_y);
                
                if (area > max_area) {
                    max_area = area;
                    best_label = current_label;
                    best_min_x = min_x;
                    best_max_x = max_x;
                    best_min_y = min_y;
                    best_max_y = max_y;
                }
                current_label++;
            }
        }
    }
    
    // Create output mask
    GrayImage *mask = gray_image_create(w, h);
    for (int i = 0; i < w * h; i++) {
        mask->data[i] = (labels[i] == best_label) ? 255 : 0;
    }
    
    free(labels);
    
    if (bbox_x) *bbox_x = best_min_x;
    if (bbox_y) *bbox_y = best_min_y;
    if (bbox_w) *bbox_w = best_max_x - best_min_x + 1;
    if (bbox_h) *bbox_h = best_max_y - best_min_y + 1;
    
    return mask;
}

// Find corners of the blob
// We scan the blob to find pixels that maximize/minimize x+y and x-y
static void find_blob_corners(const GrayImage *blob, Quad *quad) {
    int w = blob->width;
    int h = blob->height;
    
    // Initialize with center to be safe
    Point2D tl = {w/2, h/2}, tr = {w/2, h/2}, br = {w/2, h/2}, bl = {w/2, h/2};
    
    float min_sum = FLT_MAX, max_sum = -FLT_MAX;
    float min_diff = FLT_MAX, max_diff = -FLT_MAX;
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (blob->data[y * w + x] > 128) {
                float sum = x + y;
                float diff = y - x;
                
                if (sum < min_sum) { min_sum = sum; tl = (Point2D){x, y}; }
                if (sum > max_sum) { max_sum = sum; br = (Point2D){x, y}; }
                if (diff < min_diff) { min_diff = diff; tr = (Point2D){x, y}; }
                if (diff > max_diff) { max_diff = diff; bl = (Point2D){x, y}; }
            }
        }
    }
    
    quad->corners[0] = tl;
    quad->corners[1] = tr;
    quad->corners[2] = br;
    quad->corners[3] = bl;
}

// ============================================================================
// QUAD DETECTION
// ============================================================================

bool find_largest_quad(const GrayImage *edges, Quad *quad) {
    // Strategy 1: Largest Connected Component (Blob)
    // This is robust for Sudoku grids which are usually the largest connected object
    // especially after dilation.
    
    int bx, by, bw, bh;
    GrayImage *blob = find_largest_blob(edges, &bx, &by, &bw, &bh);
    
    // Check if blob is large enough (at least 1/16 of image area)
    int img_area = edges->width * edges->height;
    int blob_bbox_area = bw * bh;
    
    if (blob_bbox_area > img_area / 16) {
        find_blob_corners(blob, quad);
        order_quad_corners(quad);
        gray_image_free(blob);
        return true;
    }
    
    gray_image_free(blob);
    
    // Fallback to Hough Transform if blob is too small
    // ...existing code...
    int num_lines = 0;
    // Lower threshold to detect faint lines, we will filter later
    HoughLine *lines = hough_lines(edges, 50, &num_lines);
    if (!lines || num_lines < 4) {
        free(lines);
        return false;
    }

    // Separate into horizontal and vertical lines
    // Horizontal: theta near 90 (pi/2) -> sin(theta) ~ 1, cos(theta) ~ 0
    // Vertical: theta near 0 or 180 (pi) -> sin(theta) ~ 0, cos(theta) ~ 1
    // Actually in Hough:
    // theta=0 -> normal is horizontal -> line is vertical
    // theta=90 -> normal is vertical -> line is horizontal
    
    HoughLine *horizontals = (HoughLine *)malloc(num_lines * sizeof(HoughLine));
    HoughLine *verticals = (HoughLine *)malloc(num_lines * sizeof(HoughLine));
    int h_count = 0;
    int v_count = 0;

    for (int i = 0; i < num_lines; i++) {
        float t = RAD2DEG(lines[i].theta);
        if (t > 180) t -= 180;
        
        // Vertical: theta around 0 or 180
        if (t < 30 || t > 150) {
            verticals[v_count++] = lines[i];
        }
        // Horizontal: theta around 90
        else if (t > 60 && t < 120) {
            horizontals[h_count++] = lines[i];
        }
    }

    if (h_count < 2 || v_count < 2) {
        free(lines);
        free(horizontals);
        free(verticals);
        return false;
    }

    // Find the most extreme lines to form the largest bounding box
    // We want the top-most, bottom-most, left-most, right-most
    // But we must ensure they are strong lines.
    // Since lines are sorted by votes, we can check the top N lines.
    
    int check_depth = 20; // Check top 20 lines of each type
    if (check_depth > h_count) check_depth = h_count;
    int v_check_depth = 20;
    if (v_check_depth > v_count) v_check_depth = v_count;

    Quad best_quad;
    bool found = false;

    // Brute force pairs of H and V lines to find largest area
    // This is simplified. A better way is to find the hull.
    // But for Sudoku, the grid is usually the dominant lines.
    
    // Strategy: Find Top/Bottom and Left/Right
    // Sort horizontals by rho (y-intercept approx)
    // Sort verticals by rho (x-intercept approx)
    
    // We need to keep the original vote sort for priority, but finding extremes requires rho.
    // Let's just iterate and find the pair that maximizes distance * votes?
    // No, just distance.
    
    // Let's pick the pair of H lines with max delta-rho, and pair of V lines with max delta-rho
    // FROM the top voted lines.
    
    HoughLine top = horizontals[0], bottom = horizontals[0];
    HoughLine left = verticals[0], right = verticals[0];
    
    // Initialize with extremes from the top K voted lines
    float min_rho_h = FLT_MAX, max_rho_h = -FLT_MAX;
    float min_rho_v = FLT_MAX, max_rho_v = -FLT_MAX;
    
    for (int i = 0; i < check_depth; i++) {
        if (horizontals[i].rho < min_rho_h) { min_rho_h = horizontals[i].rho; top = horizontals[i]; }
        if (horizontals[i].rho > max_rho_h) { max_rho_h = horizontals[i].rho; bottom = horizontals[i]; }
    }
    
    for (int i = 0; i < v_check_depth; i++) {
        if (verticals[i].rho < min_rho_v) { min_rho_v = verticals[i].rho; left = verticals[i]; }
        if (verticals[i].rho > max_rho_v) { max_rho_v = verticals[i].rho; right = verticals[i]; }
    }
    
    // Compute intersections
    Point2D tl = intersection(top, left);
    Point2D tr = intersection(top, right);
    Point2D bl = intersection(bottom, left);
    Point2D br = intersection(bottom, right);
    
    // Check if valid (points inside or near image)
    // Allow some margin outside
    int w = edges->width;
    int h = edges->height;
    int margin = w / 2; // Large margin
    
    if (tl.x > -margin && tl.x < w+margin && tl.y > -margin && tl.y < h+margin) {
        best_quad.corners[0] = tl;
        best_quad.corners[1] = tr;
        best_quad.corners[2] = br;
        best_quad.corners[3] = bl;
        found = true;
    }

    free(lines);
    free(horizontals);
    free(verticals);
    
    if (found) {
        *quad = best_quad;
        order_quad_corners(quad);
    }
    
    return found;
}

void order_quad_corners(Quad *quad) {
    // Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    // Based on sum (x+y) and diff (y-x)
    
    Point2D center = {0, 0};
    for (int i = 0; i < 4; i++) {
        center.x += quad->corners[i].x;
        center.y += quad->corners[i].y;
    }
    center.x /= 4;
    center.y /= 4;
    
    Point2D ordered[4];
    
    for (int i = 0; i < 4; i++) {
        Point2D p = quad->corners[i];
        if (p.x < center.x && p.y < center.y) ordered[0] = p; // TL
        else if (p.x > center.x && p.y < center.y) ordered[1] = p; // TR
        else if (p.x > center.x && p.y > center.y) ordered[2] = p; // BR
        else if (p.x < center.x && p.y > center.y) ordered[3] = p; // BL
    }
    
    // Fallback if simple quadrant check fails (e.g. rotated)
    // Sort by Y then X?
    // Standard method:
    // TL: min(x+y)
    // BR: max(x+y)
    // TR: min(y-x) ? No, max(x-y) -> x large, y small
    // BL: max(y-x) ? No, min(x-y) -> x small, y large
    
    // Let's use the sum/diff method which is robust
    int min_sum_idx = 0;
    int max_sum_idx = 0;
    int min_diff_idx = 0;
    int max_diff_idx = 0;
    
    float min_sum = FLT_MAX, max_sum = -FLT_MAX;
    float min_diff = FLT_MAX, max_diff = -FLT_MAX;
    
    for (int i = 0; i < 4; i++) {
        float sum = quad->corners[i].x + quad->corners[i].y;
        float diff = quad->corners[i].y - quad->corners[i].x;
        
        if (sum < min_sum) { min_sum = sum; min_sum_idx = i; }
        if (sum > max_sum) { max_sum = sum; max_sum_idx = i; }
        if (diff < min_diff) { min_diff = diff; min_diff_idx = i; }
        if (diff > max_diff) { max_diff = diff; max_diff_idx = i; }
    }
    
    ordered[0] = quad->corners[min_sum_idx]; // TL
    ordered[1] = quad->corners[min_diff_idx]; // TR
    ordered[2] = quad->corners[max_sum_idx]; // BR
    ordered[3] = quad->corners[max_diff_idx]; // BL
    
    for(int i=0; i<4; i++) quad->corners[i] = ordered[i];
}

