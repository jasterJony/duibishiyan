#!/usr/bin/env python3
"""Convert YOLO format dataset to COCO format for MMDetection."""

import json
import os
from PIL import Image
from pathlib import Path

# COCO 80 classes
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def convert_yolo_to_coco(yolo_dir, output_dir, split='train'):
    """Convert YOLO format to COCO format.
    
    Args:
        yolo_dir: Path to YOLO dataset (contains images/ and labels/)
        output_dir: Output directory for COCO format
        split: 'train' or 'val'
    """
    yolo_dir = Path(yolo_dir)
    output_dir = Path(output_dir)
    
    images_dir = yolo_dir / 'images' / split
    labels_dir = yolo_dir / 'labels' / split
    
    # Create output directories
    out_images_dir = output_dir / f'{split}2017'
    out_ann_dir = output_dir / 'annotations'
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize COCO format
    coco_format = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # Add categories
    for i, name in enumerate(COCO_CLASSES):
        coco_format['categories'].append({
            'id': i,
            'name': name,
            'supercategory': 'object'
        })
    
    annotation_id = 1
    
    # Process each image
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    for img_id, img_path in enumerate(sorted(image_files), start=1):
        # Get image info
        img = Image.open(img_path)
        width, height = img.size
        
        # Copy/link image to output directory
        out_img_path = out_images_dir / img_path.name
        if not out_img_path.exists():
            import shutil
            shutil.copy(img_path, out_img_path)
        
        # Add image info
        coco_format['images'].append({
            'id': img_id,
            'file_name': img_path.name,
            'width': width,
            'height': height
        })
        
        # Read corresponding label file
        label_path = labels_dir / (img_path.stem + '.txt')
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bbox_width = float(parts[3])
                        bbox_height = float(parts[4])
                        
                        # Convert YOLO format (normalized center x, y, w, h) to COCO format (x, y, w, h in pixels)
                        x = (x_center - bbox_width / 2) * width
                        y = (y_center - bbox_height / 2) * height
                        w = bbox_width * width
                        h = bbox_height * height
                        
                        # Add annotation
                        coco_format['annotations'].append({
                            'id': annotation_id,
                            'image_id': img_id,
                            'category_id': class_id,
                            'bbox': [x, y, w, h],
                            'area': w * h,
                            'iscrowd': 0,
                            'segmentation': []  # Empty for detection only
                        })
                        annotation_id += 1
    
    # Save annotation file
    ann_file = out_ann_dir / f'instances_{split}2017.json'
    with open(ann_file, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f'Converted {split}: {len(coco_format["images"])} images, {len(coco_format["annotations"])} annotations')
    print(f'Saved to: {ann_file}')


def main():
    yolo_dir = 'data/coco8'
    output_dir = 'data/coco8'
    
    # Convert train and val splits
    convert_yolo_to_coco(yolo_dir, output_dir, 'train')
    convert_yolo_to_coco(yolo_dir, output_dir, 'val')
    
    print('\nConversion completed!')
    print('Dataset structure:')
    print('  data/coco8/')
    print('  ├── annotations/')
    print('  │   ├── instances_train2017.json')
    print('  │   └── instances_val2017.json')
    print('  ├── train2017/')
    print('  └── val2017/')


if __name__ == '__main__':
    main()
