import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random

def generate_digit_dataset(num_images=10000, output_dir='digital_digits_dataset'):
    """
    Génère un dataset de chiffres digitaux en 28x28 pixels, blanc sur fond noir.
    Compatible avec MNIST pour la data augmentation.
    
    Args:
        num_images: Nombre d'images à générer (défaut: 10000)
        output_dir: Dossier de sortie pour les images
    """
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Polices courantes disponibles sur la plupart des systèmes
    # On ne garde que celles où le '1' est un simple bâton (sans barre en bas)
    # pour éviter la confusion avec le '7' ou d'autres artefacts
    # Note: Sur Linux, il faut souvent le chemin complet
    font_candidates = [
        # Microsoft fonts (si installées)
        'arial.ttf', 'Arial.ttf',
        'verdana.ttf', 'Verdana.ttf',
        'tahoma.ttf', 'Tahoma.ttf',
        # Linux standard fonts (DejaVu Sans est très proche de Arial)
        # '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        # '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        # 'DejaVuSans.ttf', 'DejaVuSans-Bold.ttf'
    ]
    
    # Filtrer les polices qui existent réellement sur le système
    available_fonts = []
    for f in font_candidates:
        try:
            # Test de chargement
            ImageFont.truetype(f, 20)
            available_fonts.append(f)
        except:
            continue
            
    if not available_fonts:
        print("ATTENTION: Aucune police spécifique trouvée. Utilisation de la police par défaut (très basique).")
        print("Installez ttf-mscorefonts-installer ou utilisez les polices DejaVu.")
    else:
        print(f"Polices trouvées et utilisées: {available_fonts}")
    # Tailles de police à utiliser (varie pour plus de diversité)
    font_sizes = list(range(18, 26))
    
    # Chiffres à générer
    digits = list(range(10))
    
    # Stockage des labels
    labels = []
    
    print(f"Génération de {num_images} images de chiffres digitaux...")
    # Ouvrir le fichier metadata pour enregistrer quel font a été utilisé par image
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    meta_file = open(metadata_path, 'w', encoding='utf-8')
    meta_file.write('filename,label,font\n')
    
    for img_idx in range(num_images):
        # Choisir un chiffre aléatoire
        digit = random.choice(digits)
        labels.append(digit)
        
        # Créer une image noire
        img = Image.new('L', (28, 28), color=0)
        draw = ImageDraw.Draw(img)
        
        # Essayer de charger une police aléatoire
        font = None
        font_size = random.choice(font_sizes)
        resolved_font_name = 'default'
        
        if available_fonts:
            font_name = random.choice(available_fonts)
            try:
                font = ImageFont.truetype(font_name, font_size)
                resolved_font_name = os.path.basename(font_name)
            except:
                pass # Fallback to default below
        
        # Si aucune police trouvée, utiliser la police par défaut
        if font is None:
            font = ImageFont.load_default()
            resolved_font_name = 'PIL_default'
        
        # Obtenir la taille du texte
        text = str(digit)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Ajouter un peu de variation dans la position (centrage avec léger décalage aléatoire)
        offset_x = random.randint(-2, 2)
        offset_y = random.randint(-2, 2)
        
        x = (28 - text_width) // 2 + offset_x
        y = (28 - text_height) // 2 + offset_y
        
        # Dessiner le chiffre en blanc
        draw.text((x, y), text, fill=255, font=font)
        
        # Ajouter une légère rotation aléatoire (-10 à +10 degrés)
        # On utilise resample=Image.BILINEAR pour garder une bonne qualité
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        
        # Optionnel: ajouter un léger bruit ou flou pour plus de réalisme
        if random.random() < 0.3:  # 30% de chance d'ajouter un léger flou
            from PIL import ImageFilter
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Sauvegarder l'image
        filename = f'digit_{img_idx:05d}_label_{digit}.png'
        img.save(os.path.join(output_dir, filename))
        
        # Enregistrer les métadonnées
        meta_file.write(f'{filename},{digit},{resolved_font_name}\n')
        
        # Afficher la progression
        if (img_idx + 1) % 1000 == 0:
            print(f"  {img_idx + 1}/{num_images} images générées")
    
    meta_file.close()
    
    # Sauvegarder les labels dans un fichier
    labels_array = np.array(labels)
    np.save(os.path.join(output_dir, 'labels.npy'), labels_array)
    
    print(f"\n✓ Génération terminée!")
    print(f"  - Images sauvegardées dans: {output_dir}/")
    print(f"  - Labels sauvegardés dans: {output_dir}/labels.npy")
    print(f"  - Distribution des classes:")
    
    unique, counts = np.unique(labels_array, return_counts=True)
    for digit, count in zip(unique, counts):
        print(f"    Chiffre {digit}: {count} images")

def load_dataset(data_dir='digital_digits_dataset'):
    """
    Charge le dataset généré.
    
    Returns:
        images: array numpy de forme (N, 28, 28)
        labels: array numpy de forme (N,)
    """
    labels = np.load(os.path.join(data_dir, 'labels.npy'))
    
    images = []
    for img_file in sorted(os.listdir(data_dir)):
        if img_file.endswith('.png'):
            img_path = os.path.join(data_dir, img_file)
            img = Image.open(img_path)
            images.append(np.array(img))
    
    return np.array(images), labels

# Exemple d'utilisation
if __name__ == "__main__":
    # Générer le dataset
    generate_digit_dataset(num_images=60000, output_dir='../digital_digits_dataset')
    
    # Exemple de chargement
    print("\nTest de chargement du dataset...")
    images, labels = load_dataset('../digital_digits_dataset')
    print(f"Dataset chargé: {images.shape}, labels: {labels.shape}")
    
    # Afficher quelques statistiques
    print(f"Valeur min des pixels: {images.min()}")
    print(f"Valeur max des pixels: {images.max()}")
    print(f"Type de données: {images.dtype}")