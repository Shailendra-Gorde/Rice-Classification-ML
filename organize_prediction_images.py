"""
Organize prediction history images into training folders
This helps you use your prediction images for training
"""

import os
import json
import shutil
from collections import defaultdict

print("="*70)
print("ORGANIZING PREDICTION IMAGES FOR TRAINING")
print("="*70)

# Configuration
PREDICTION_HISTORY_FILE = 'results/prediction_history.json'
UPLOAD_FOLDER = 'uploads'
TRAINING_IMAGES_DIR = 'data/images'

# Create training images directory structure
os.makedirs(TRAINING_IMAGES_DIR, exist_ok=True)

# Load prediction history
if not os.path.exists(PREDICTION_HISTORY_FILE):
    print(f"\n[ERROR] Prediction history not found: {PREDICTION_HISTORY_FILE}")
    print("        Make some predictions first, then run this script.")
    exit(1)

print(f"\nLoading prediction history from: {PREDICTION_HISTORY_FILE}")
with open(PREDICTION_HISTORY_FILE, 'r') as f:
    history = json.load(f)

predictions = history.get('predictions', [])
print(f"Found {len(predictions)} predictions")

if len(predictions) == 0:
    print("\n[ERROR] No predictions found in history.")
    print("        Make some predictions first, then run this script.")
    exit(1)

# Group predictions by variety
variety_predictions = defaultdict(list)
for pred in predictions:
    variety = pred.get('variety', 'Unknown')
    if variety and variety != 'Unknown':
        variety_predictions[variety].append(pred)

print(f"\nPredictions by variety:")
for variety, preds in variety_predictions.items():
    print(f"  {variety}: {len(preds)} predictions")

# Create variety folders and copy images
copied_count = 0
skipped_count = 0

print(f"\nOrganizing images into: {TRAINING_IMAGES_DIR}")

for variety, preds in variety_predictions.items():
    # Create variety folder
    variety_dir = os.path.join(TRAINING_IMAGES_DIR, variety)
    os.makedirs(variety_dir, exist_ok=True)
    
    print(f"\n{variety}:")
    
    for pred in preds:
        image_filename = pred.get('image_filename')
        if not image_filename:
            skipped_count += 1
            continue
        
        # Source path
        source_path = os.path.join(UPLOAD_FOLDER, image_filename)
        
        if not os.path.exists(source_path):
            skipped_count += 1
            continue
        
        # Destination path (use original filename or create unique name)
        dest_filename = image_filename
        dest_path = os.path.join(variety_dir, dest_filename)
        
        # If file already exists, add counter
        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(image_filename)
            dest_filename = f"{name}_{counter}{ext}"
            dest_path = os.path.join(variety_dir, dest_filename)
            counter += 1
        
        try:
            shutil.copy2(source_path, dest_path)
            copied_count += 1
            print(f"  ✓ Copied: {dest_filename}")
        except Exception as e:
            print(f"  ✗ Failed to copy {image_filename}: {e}")
            skipped_count += 1

print("\n" + "="*70)
print("ORGANIZATION COMPLETE!")
print("="*70)
print(f"\n✓ Copied {copied_count} images")
if skipped_count > 0:
    print(f"⚠ Skipped {skipped_count} images (not found or errors)")

print(f"\n📁 Images organized in: {TRAINING_IMAGES_DIR}")
print("\n📝 Next Steps:")
print("1. Review the organized images")
print("2. Add more images to each variety folder if needed")
print("3. Run: python train_with_real_images.py")
print("\n💡 Tip: More images per variety = Better accuracy!")
print("="*70)
