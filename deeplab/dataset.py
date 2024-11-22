"""
CS 5330
Final Project: Garbage Classification
Group members: Jinghan Gao, Tianhao Zhang, Jialu Bi

This file defines the dataset to be used for training.
"""

import os
import json
import numpy as np
from PIL import Image, ExifTags
import torch
from torch.utils.data import Dataset
from skimage.draw import polygon as draw_polygon
from torchvision.transforms.functional import resize

class TacoDataset(Dataset):
    def __init__(self, dataset_dir, class_map=None, transform=None):
        """
        Args:
            dataset_dir (str): Path to the dataset directory containing images and annotations.
            class_map (dict): Mapping of original class names to new class names.
            transform (callable, optional): Transform to apply to images and masks.
        """
        self.dataset_dir = dataset_dir
        self.transform = transform
        
        # Load the annotations
        ann_filepath = os.path.join(dataset_dir, 'annotations.json')
        assert os.path.isfile(ann_filepath), "Annotations file not found."
        with open(ann_filepath, 'r') as f:
            self.dataset = json.load(f)
        
        # Replace dataset classes if class_map is provided
        if class_map:
            self.replace_dataset_classes(class_map)
        
        # Build image and annotation lists
        self.images = {img['id']: img for img in self.dataset['images']}
        self.annotations = self.dataset['annotations']
        self.class_map = {cat['id']: cat['name'] for cat in self.dataset['categories']}
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): The input image.
            mask (Tensor): The segmentation mask.
        """
        img_id = list(self.images.keys())[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.dataset_dir, img_info['file_name'])
        
        # Load the image
        image = Image.open(img_path)
        # Load metadata
        exif = image._getexif()
        if exif:
            exif = dict(exif.items())
            # Rotate portrait images if necessary (274 is the orientation tag code)
            if 274 in exif:
                if exif[274] == 3:
                    image = image.rotate(180, expand=True)
                if exif[274] == 6:
                    image = image.rotate(270, expand=True)
                if exif[274] == 8:
                    image = image.rotate(90, expand=True)
        
        # Create the segmentation mask
        mask = self.create_mask(img_id, img_info['height'], img_info['width'])
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)

            mask = mask.unsqueeze(0)
            mask = resize(mask, (256, 256), interpolation=Image.NEAREST)
        
        return image, mask.unsqueeze(0).squeeze(0).squeeze(0)
    
    def create_mask(self, img_id, height, width):
        """
        Create a segmentation mask for the given image ID.
        Args:
            img_id (int): ID of the image.
            height (int): Height of the mask.
            width (int): Width of the mask.
        Returns:
            mask (Tensor): Segmentation mask with class IDs.
        """
        mask = np.zeros((height, width), dtype=np.int32)
        
        for ann in self.annotations:
            if ann['image_id'] == img_id:
                category_id = ann['category_id']
                segmentation = ann['segmentation']
                # Convert segmentation polygons to a mask
                for poly in segmentation:
                    coords = np.array(poly).reshape((-1, 2)).astype(np.int32)
                    poly_mask = self.poly_to_mask(coords, height, width)
                    mask[poly_mask > 0] = category_id
        
        return torch.tensor(mask, dtype=torch.long)
    
    @staticmethod
    def poly_to_mask(polygon, height, width):
        """
        Convert a polygon to a binary mask.
        Args:
            polygon (np.array): Nx2 array of (x, y) coordinates.
            height (int): Mask height.
            width (int): Mask width.
        Returns:
            mask (np.array): Binary mask.
        """
        rr, cc = draw_polygon(polygon[:, 1], polygon[:, 0], shape=(height, width))
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[rr, cc] = 1
        return mask
    
    def replace_dataset_classes(self, class_map):
        """Replace classes in the dataset with new classes based on the class_map."""
        class_new_names = list(set(class_map.values()))
        class_new_names.sort()
        class_originals = self.dataset['categories']
        self.dataset['categories'] = []
        class_ids_map = {}  # Map from old ID to new ID
        
        # Replace categories
        for id_new, class_new_name in enumerate(class_new_names):
            category = {
                'id': id_new,
                'name': class_new_name,
            }
            self.dataset['categories'].append(category)
            # Map class names
            for class_original in class_originals:
                if class_map[class_original['name']] == class_new_name:
                    class_ids_map[class_original['id']] = id_new
        
        # Update annotations
        for ann in self.dataset['annotations']:
            ann['category_id'] = class_ids_map[ann['category_id']]

