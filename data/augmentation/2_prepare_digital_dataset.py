import os
import numpy as np
from PIL import Image
import struct
import random
import glob

DATASET_DIR = "../digital_digits_dataset"
OUTPUT_TRAIN = "../digital_train.bin"
OUTPUT_TEST = "../digital_test.bin"
IMG_SIZE = 28
MAGIC_NUMBER = 0xDEADBEEF

def load_images_from_folder(folder):
    images = []
    labels = []
    
    # Pattern to match files like digit_05994_label_9.png
    file_pattern = os.path.join(folder, "digit_*_label_*.png")
    files = glob.glob(file_pattern)
    
    print(f"Found {len(files)} images in {folder}")
    
    for filepath in files:
        try:
            # Extract label from filename
            filename = os.path.basename(filepath)
            # digit_XXXXX_label_Y.png -> Y
            label = int(filename.split('_label_')[1].split('.')[0])
            
            # Load image
            img = Image.open(filepath).convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_data = np.array(img, dtype=np.uint8)
            
            images.append(img_data)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
    return images, labels

def save_to_binary(filename, images, labels):
    count = len(images)
    if count == 0:
        print(f"No images to save to {filename}")
        return

    print(f"Saving {count} images to {filename}...")
    
    with open(filename, 'wb') as f:
        # Header
        f.write(struct.pack('>I', MAGIC_NUMBER)) # Big endian magic
        f.write(struct.pack('>I', count))
        f.write(struct.pack('>I', IMG_SIZE))
        f.write(struct.pack('>I', IMG_SIZE))
        
        # Data
        for img, label in zip(images, labels):
            f.write(struct.pack('B', label))
            f.write(img.tobytes())

def main():
    images, labels = load_images_from_folder(DATASET_DIR)
    
    if not images:
        print("No images found!")
        return

    # Shuffle
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images[:], labels[:] = zip(*combined)
    
    # Split 70/30
    split_idx = int(len(images) * 0.7)
    
    train_images = images[:split_idx]
    train_labels = labels[:split_idx]
    
    test_images = images[split_idx:]
    test_labels = labels[split_idx:]
    
    print(f"Split: {len(train_images)} train, {len(test_images)} test")
    
    save_to_binary(OUTPUT_TRAIN, train_images, train_labels)
    save_to_binary(OUTPUT_TEST, test_images, test_labels)
    print("Done.")

if __name__ == "__main__":
    main()
