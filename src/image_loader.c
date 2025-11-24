#include "image_loader.h"
#include <stdio.h>
#include <stdlib.h>

// Implémentation STB (header-only libraries)
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image.h"
#include "external/stb_image_write.h"

// ============================================================================
// CHARGEMENT D'IMAGES
// ============================================================================

RGBImage* load_rgb_image(const char *filename) {
    int width, height, channels;
    
    // Force le chargement en RGB (3 canaux)
    unsigned char *data = stbi_load(filename, &width, &height, &channels, 3);
    
    if (!data) {
        LOG_ERROR("Impossible de charger l'image: %s", filename);
        return NULL;
    }
    
    RGBImage *img = rgb_image_create(width, height, 3);
    if (!img) {
        stbi_image_free(data);
        return NULL;
    }
    
    // Copier les données
    size_t total_bytes = width * height * 3;
    for (size_t i = 0; i < total_bytes; i++) {
        img->data[i] = data[i];
    }
    
    stbi_image_free(data);
    
    LOG_INFO("Image chargée: %s (%dx%d, %d canaux)", filename, width, height, 3);
    return img;
}

GrayImage* load_gray_image(const char *filename) {
    int width, height, channels;
    
    // Force le chargement en niveaux de gris (1 canal)
    unsigned char *data = stbi_load(filename, &width, &height, &channels, 1);
    
    if (!data) {
        LOG_ERROR("Impossible de charger l'image: %s", filename);
        return NULL;
    }
    
    GrayImage *img = gray_image_create(width, height);
    if (!img) {
        stbi_image_free(data);
        return NULL;
    }
    
    // Copier les données
    size_t total_bytes = width * height;
    for (size_t i = 0; i < total_bytes; i++) {
        img->data[i] = data[i];
    }
    
    stbi_image_free(data);
    
    LOG_INFO("Image en niveaux de gris chargée: %s (%dx%d)", filename, width, height);
    return img;
}

// ============================================================================
// SAUVEGARDE D'IMAGES
// ============================================================================

bool save_rgb_image(const char *filename, const RGBImage *img) {
    int result = stbi_write_png(filename, img->width, img->height, 
                                 img->channels, img->data, 
                                 img->width * img->channels);
    
    if (result) {
        LOG_INFO("Image RGB sauvegardée: %s", filename);
        return true;
    } else {
        LOG_ERROR("Échec sauvegarde: %s", filename);
        return false;
    }
}

bool save_gray_image(const char *filename, const GrayImage *img) {
    int result = stbi_write_png(filename, img->width, img->height, 
                                 1, img->data, img->width);
    
    if (result) {
        LOG_INFO("Image en niveaux de gris sauvegardée: %s", filename);
        return true;
    } else {
        LOG_ERROR("Échec sauvegarde: %s", filename);
        return false;
    }
}

// ============================================================================
// CONVERSIONS
// ============================================================================

GrayImage* rgb_to_gray(const RGBImage *rgb) {
    GrayImage *gray = gray_image_create(rgb->width, rgb->height);
    if (!gray) return NULL;
    
    // Formule standard de conversion RGB -> Grayscale
    // Gray = 0.299*R + 0.587*G + 0.114*B
    for (size_t i = 0; i < rgb->width * rgb->height; i++) {
        size_t rgb_idx = i * rgb->channels;
        float r = rgb->data[rgb_idx];
        float g = rgb->data[rgb_idx + 1];
        float b = rgb->data[rgb_idx + 2];
        
        gray->data[i] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
    }
    
    return gray;
}

RGBImage* gray_to_rgb(const GrayImage *gray) {
    RGBImage *rgb = rgb_image_create(gray->width, gray->height, 3);
    if (!rgb) return NULL;
    
    // Copier la valeur de gris sur les 3 canaux RGB
    for (size_t i = 0; i < gray->width * gray->height; i++) {
        uint8_t val = gray->data[i];
        rgb->data[i * 3] = val;
        rgb->data[i * 3 + 1] = val;
        rgb->data[i * 3 + 2] = val;
    }
    
    return rgb;
}
