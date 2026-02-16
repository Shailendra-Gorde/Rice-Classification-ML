"""
Setup script to create training folder structure for all 13 rice varieties
This helps organize your training images
"""

import os
from src.image_preprocessing import INDIAN_RICE_VARIETIES

print("="*70)
print("SETTING UP TRAINING FOLDERS FOR ALL 13 RICE VARIETIES")
print("="*70)

IMAGES_DIR = 'data/images'

# Create main directory
os.makedirs(IMAGES_DIR, exist_ok=True)

print(f"\nCreating folders in: {IMAGES_DIR}\n")

# Create folder for each variety
created_folders = []
existing_folders = []

for variety in INDIAN_RICE_VARIETIES:
    variety_dir = os.path.join(IMAGES_DIR, variety)
    
    if os.path.exists(variety_dir):
        existing_folders.append(variety)
        print(f"  ✓ {variety} (already exists)")
    else:
        os.makedirs(variety_dir, exist_ok=True)
        created_folders.append(variety)
        print(f"  ✓ {variety} (created)")

print("\n" + "="*70)
print("FOLDER SETUP COMPLETE!")
print("="*70)

if created_folders:
    print(f"\n✓ Created {len(created_folders)} new folders")
if existing_folders:
    print(f"✓ Found {len(existing_folders)} existing folders")

print(f"\n📁 All 13 variety folders are ready in: {IMAGES_DIR}/")
print("\n📝 Next Steps:")
print("1. Add images to each variety folder:")
print("   - Minimum: 10 images per variety")
print("   - Recommended: 50-100+ images per variety")
print("   - Ideal: 100-200+ images per variety")
print("\n2. Organize your images:")
print(f"   {IMAGES_DIR}/")
for variety in INDIAN_RICE_VARIETIES:
    print(f"   ├── {variety}/")
    print(f"   │   ├── image1.jpg")
    print(f"   │   ├── image2.jpg")
    print(f"   │   └── ... (add your images here)")
print("\n3. After adding images, run:")
print("   python train_all_13_varieties.py")
print("="*70)
