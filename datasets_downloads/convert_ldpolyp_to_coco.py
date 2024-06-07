#!/usr/bin/env python3
# Copyright (C) 2024 Cosmo Intelligent Medical Devices
#
#
# Portions of this file are derived from the EndoCV2021-polyp_det_seg_gen project:
# https://github.com/sharib-vision/EndoCV2021-polyp_det_seg_gen
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
import json
import cv2


def detect_imgs(infolder, ext='.tif'):
    """
    Lists sorted paths to image files in a directory with a specified extension.

    Args:
        infolder (str): Directory containing the images.
        ext (str): Extension of images to find.

    Returns:
        ndarray: Sorted array of image file paths.
    """

    items = os.listdir(infolder)

    # Add image paths into flist and return
    flist = []
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))

    return np.sort(flist)


def coco_annotation_format(image_id, category_id, bbox, annotation_id, image_width, image_height):
    """
    Formats a single annotation in COCO style.

    Args:
        image_id (int): ID of the image.
        category_id (int): Category ID of the object.
        bbox (list): Bounding box [x_min, y_min, x_max, y_max].
        annotation_id (int): Unique annotation ID.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        dict: Formatted annotation.
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
    Generates a COCO-formatted dataset from images and corresponding annotation files.

    Args:
        all_img_list (list): List of all image file paths.
        coco_folder (str): Path to the folder containing annotation files.

    Returns:
        dict: COCO dataset including images, annotations, and category specification.
    """

    # Create the dictionary for the coco annotations
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "polyp"}]
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
        bboxes = []
        ann_file = f"Annotations/{filename.split('.')[0]}.txt"
        with open(os.path.join(coco_folder, ann_file), "r") as file:
            for line in file:
                line = line.split()  # to deal with blank
                if line:  # lines (ie skip them)
                    line = [int(i) for i in line]
                    bboxes.append(line)

            # Format the data
            annotations = []
            for i in range(bboxes[0][0]):
                annotations.append(coco_annotation_format(image_id, 1, bboxes[i+1],
                                                          annotation_id, image_width, image_height))
                annotation_id += 1

        coco_data["annotations"].extend(annotations)

        # Increment image ID
        image_id += 1

    return coco_data


def convert(base_folder):
    """
    Converts a set of images and annotation files into a COCO-formatted JSON file.

    Args:
        base_folder (str): Base directory to find and save COCO data.

    Returns:
        None: A JSON file is generated with the COCO-formatted dataset.
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
