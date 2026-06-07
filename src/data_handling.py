"""
data_handling.py — Normalise raw image files into clean JPEGs.

Converts every image in the dataset to RGB JPEG format and renames files to
a consistent pattern: {split}_{class}_{index:04d}.jpg

Run this ONCE before training, on your raw downloaded images.

Usage:
    python src/data_handling.py
    python src/data_handling.py --data_root /path/to/raw/data
"""

import argparse
import os
from PIL import Image

# Project root = one level above this file (src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def process_split(root: str, split: str) -> None:
    split_path = os.path.join(root, split)
    if not os.path.isdir(split_path):
        print(f'[!] Skipping {split_path} — folder not found')
        return

    for class_name in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        print(f'[→] {split}/{class_name}: {len(files)} files found')

        converted = 0
        errors    = 0

        for i, filename in enumerate(sorted(files)):
            src_path = os.path.join(class_path, filename)
            dst_path = os.path.join(class_path, f'{split}_{class_name}_{i:04d}.jpg')

            try:
                img = Image.open(src_path).convert('RGB')
                img.save(dst_path, format='JPEG', quality=95)

                if src_path != dst_path:
                    os.remove(src_path)

                converted += 1

            except Exception as e:
                print(f'    [!] Error on {filename}: {e}')
                errors += 1

        print(f'    ✓ {converted} converted, {errors} errors')


def main(args):
    print(f'[✓] Processing dataset at: {args.data_root}\n')
    for split in ['train', 'val', 'test']:
        process_split(args.data_root, split)
    print('\n[✓] Done. Your dataset is ready for training.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Normalise dataset images to RGB JPEG')
    parser.add_argument('--data_root', default=os.path.join(PROJECT_ROOT, 'data'), help='Root folder of the dataset')
    args = parser.parse_args()
    main(args)