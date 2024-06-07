#!/usr/bin/env python3
# Copyright (C) 2024 Cosmo Intelligent Medical Devices
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import os
import xml.etree.ElementTree as ET
import cv2
import json


def detect_imgs(infolder, ext='.tif'):
    """
    Detects and sorts image files with a specific extension within a directory.

    Args:
        infolder (str): The directory to search for image files.
        ext (str): The file extension of images to detect.

    Returns:
        ndarray: An array of sorted file paths.
    """

    items = os.listdir(infolder)

    flist = []

    # Add image paths into list and return
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))

    return np.sort(flist)


def coco_annotation_format(image_id, category_id, bbox, annotation_id, image_width, image_height):
    """
    Formats an annotation for a single object in COCO format.

    Args:
        image_id (int): The ID of the image the object is in.
        category_id (int): The category ID of the object.
        bbox (list): The bounding box of the object, as [x_min, y_min, x_max, y_max].
        annotation_id (int): The unique ID for this annotation.
        image_width (int): The width of the image.
        image_height (int): The height of the image.

    Returns:
        dict: The annotation formatted in COCO style.
    """

    bounded_x = max(0, bbox[0])
    bounded_y = max(0, bbox[1])
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [bounded_x,
                 bounded_y,
                 min(image_width - bounded_x, bbox[2] - bounded_x),
                 min(image_height - bounded_y, bbox[3] - bounded_y)],
        "area": min(image_width - bounded_x, bbox[2] - bounded_x) * min(image_height - bounded_y, bbox[3] - bounded_y),
        "iscrowd": 0
    }


def create_coco_dataset(all_img_list, coco_folder):
    """
    Generates a COCO-style dataset from a list of image files and corresponding annotation files.

    Args:
        all_img_list (list): A list of paths to image files.
        coco_folder (str): The folder containing corresponding XML annotation files.

    Returns:
        dict: A dictionary structured as a COCO dataset.
    """
    # Create the dictionary for the coco annotations
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "polyp"}]  # Assuming 'polyp' is your category
    }
    annotation_id = 1
    image_id = 1

    # Loop through each imager
    for imageFile in all_img_list:
        image = cv2.imread(imageFile)
        image_height, image_width = image.shape[:2]
        filename = os.path.basename(imageFile)

        # Fill out the images section
        coco_data["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": image_width,
            "height": image_height
        })

        # Parse the annotations file to extract the boxes
        ann_file = f"Annotations/{filename.split('.')[0]}.xml"
        tree = ET.parse(os.path.join(coco_folder, ann_file))
        root = tree.getroot()

        annotations = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            bbox = [xmin, ymin, xmax, ymax]

            # Format the data
            annotations.append(coco_annotation_format(image_id, 1, bbox,
                                                      annotation_id, image_width, image_height))
            annotation_id += 1

        coco_data["annotations"].extend(annotations)

        # Increment image ID
        image_id += 1

    return coco_data


def convert(base_folder):
    """
    Converts a dataset of images and XML annotations into COCO format and saves it as a JSON file.

    Args:
        base_folder (str): The base folder where the COCO directory will be created and the JSON file saved.

    Returns:
        None. The function creates a JSON file containing the COCO-formatted dataset.
    """

    coco_folder = os.path.join(base_folder, "coco/")
    os.makedirs(coco_folder, exist_ok=True)

    print(f"Processing Data...")
    # Define the paths for images and annotations for training and validation
    train_image_path = os.path.join(coco_folder, "train_images")

    # Detect and collect all images and masks
    all_img_list = detect_imgs(train_image_path, ext='.jpg')

    # Create COCO dataset for the current center and append it
    coco_dataset = create_coco_dataset(all_img_list, coco_folder)

    # Save to JSON file
    json_save_file = os.path.join(base_folder, 'coco/train_ann.json')
    with open(json_save_file, 'w') as f:
        json.dump(coco_dataset, f)

    print("DONE.")
