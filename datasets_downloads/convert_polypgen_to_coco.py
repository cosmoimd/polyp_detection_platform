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
import shutil
import os
from extract_polyp_boxes import get_bbox_cordinates_from_mask_coco


def detect_imgs(infolder, ext='.tif'):
    """
    Lists sorted image file paths with a given extension from a directory.

    Args:
        infolder (str): Directory to search for image files.
        ext (str): Image file extension to look for.

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


def create_dir(path):
    """
    Creates a directory if it does not already exist.

    Args:
        path (str): Path of the directory to create.

    Returns:
        None. A directory is created at the specified path.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")


def create_coco_dataset(allfileList, allMaskList):
    """
    Generates a COCO-formatted dataset from lists of image and mask files.

    Args:
        allfileList (list): List of paths to image files.
        allMaskList (list): List of paths to corresponding mask files.

    Returns:
        dict: COCO-formatted dataset.
    """

    # Define the dictionary
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "polyp"}]  # Assuming 'polyp' is your category
    }
    annotation_id = 1

    # Iterate through amd extract the annotations
    for imageFile, maskFile in zip(allfileList, allMaskList):
        image_id = annotation_id  # Incrementing annotation_id for each image
        image = cv2.imread(imageFile)
        image_height, image_width = image.shape[:2]

        im_mask = cv2.imread(maskFile, 0) > 0

        coco_data["images"].append({
            "id": image_id,
            "file_name": os.path.basename(imageFile),
            "width": image_width,
            "height": image_height
        })

        annotations, annotation_id = get_bbox_cordinates_from_mask_coco(image, im_mask, image_id, 1, annotation_id, image_width, image_height)

        coco_data["annotations"].extend(annotations)

    return coco_data


def combine_all_coco_datasets(coco_datasets):
    """
    Combines multiple COCO-formatted datasets into one.

    Args:
        coco_datasets (list): List of COCO-formatted datasets to combine.

    Returns:
        dict: A single combined COCO-formatted dataset.
    """

    combined = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "polyp"}]
    }

    next_image_id = 1
    next_annotation_id = 1

    for dataset in coco_datasets:
        # Store a mapping from the old image IDs to the new ones
        image_id_map = {}

        # Add images
        for image in dataset['images']:
            new_image = image.copy()
            old_image_id = image['id']  # Store the old image ID
            new_image['id'] = next_image_id
            image_id_map[old_image_id] = next_image_id  # Map old ID to new ID
            combined['images'].append(new_image)
            next_image_id += 1

        # Add annotations
        for annotation in dataset['annotations']:
            new_annotation = annotation.copy()

            # Update the image_id to the new image id using the mapping
            old_image_id = annotation['image_id']
            if old_image_id in image_id_map:
                new_annotation['image_id'] = image_id_map[old_image_id]
                new_annotation['id'] = next_annotation_id
                combined['annotations'].append(new_annotation)
                next_annotation_id += 1
            else:
                print("Warning: Annotation image_id does not have a corresponding image.")

    return combined


def move_images(source_directory, destination_directory, file_extension='.jpg'):
    """
    Moves image files from a source directory to a destination directory.

    Args:
        source_directory (str): Directory to move files from.
        destination_directory (str): Directory to move files to.
        file_extension (str): Extension of files to move.

    Returns:
        None. Image files are moved from the source to the destination directory.
    """

    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # loop through the directories and move the images
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if file.endswith(file_extension):
                shutil.move(os.path.join(root, file), destination_directory)


if __name__ == "__main__":
    import cv2
    import json
    'single frame data folders'

    # Updated SUBDIR_LIST to include all centers
    SUBDIR_LIST = ['data_C1', 'data_C2', 'data_C3', 'data_C4', 'data_C5', 'data_C6']

    'Sequence data folder centerwise'
    BaseFolder = '/path/to/PolyGenDataset/'

    # Define destination directory
    destination_dir = os.path.join(BaseFolder, 'coco/train_images/')
    os.makedirs(destination_dir, exist_ok=True)

    # Initialize an empty list to hold all data
    all_coco_datasets = []

    for subdir in SUBDIR_LIST:
        print(f"Converting {subdir}...")
        # Define the paths for images and masks for the current center
        allImageList = os.path.join(BaseFolder, subdir, 'images_' + subdir.split("_")[1])
        allMaskList = os.path.join(BaseFolder, subdir, 'masks_' + subdir.split("_")[1])

        # Define and create directories for bounding boxes
        bbox_C = os.path.join(BaseFolder, subdir, 'bbox_' + subdir.split("_")[1])
        bbox_image = os.path.join(BaseFolder, subdir, 'bbox_image_' + subdir.split("_")[1])
        create_dir(bbox_C)
        create_dir(bbox_image)

        # Detect and collect all images and masks
        allfileList = detect_imgs(allImageList, ext='.jpg')
        allmaskList = detect_imgs(allMaskList, ext='.jpg')

        # Create COCO dataset for the current center and append it
        coco_dataset = create_coco_dataset(allfileList, allmaskList)
        all_coco_datasets.append(coco_dataset)  # Accumulate the results

    # Process Sequence Data
    SEQUENCE_DIR = os.path.join(BaseFolder, 'sequenceData/positive/')

    for seq_num in range(1, 24):
        seq_folder = os.path.join(SEQUENCE_DIR, f'seq{seq_num}')
        print(f"Converting seq{seq_num}...")

        allImageList = os.path.join(seq_folder, f'images_seq{seq_num}')
        allMaskList = os.path.join(seq_folder, f'masks_seq{seq_num}')

        allfileList = detect_imgs(allImageList, ext='.jpg')
        allmaskList = detect_imgs(allMaskList, ext='.jpg')

        coco_dataset_for_sequence = create_coco_dataset(allfileList, allmaskList)
        all_coco_datasets.append(coco_dataset_for_sequence)

    # Process Sequence Data (Negative Only)
    NEGATIVE_SEQUENCE_DIR = os.path.join(BaseFolder, 'sequenceData/negativeOnly/')

    for seq_num in range(1, 24):  # Assuming sequences are numbered from 1 to 23
        allImageList = os.path.join(NEGATIVE_SEQUENCE_DIR, f'seq{seq_num}_neg')
        print(f"Converting seq{seq_num}_neg")

        allfileList = detect_imgs(allImageList, ext='.jpg')

        # Create COCO dataset for the negative sequence
        for imageFile in allfileList:
            image_id = len(all_coco_datasets) + 1  # Unique ID for each image
            image = cv2.imread(imageFile)
            image_height, image_width = image.shape[:2]

            all_coco_datasets.append({
                "images": [{
                    "id": image_id,
                    "file_name": os.path.basename(imageFile),
                    "width": image_width,
                    "height": image_height
                }],
                "annotations": [],
                "categories": [{"id": 1, "name": "polyp"}]
            })

    # After accumulating all data, combine them as needed (this depends on your COCO dataset structure)
    print("Combining the data...")
    combined_coco_dataset = combine_all_coco_datasets(all_coco_datasets)

    # Save to JSON file
    json_save_file = os.path.join(BaseFolder, 'coco/train_ann.json')
    with open(json_save_file, 'w') as f:
        json.dump(combined_coco_dataset, f)

    print("Moving images to coco folder...")
    # Define source directories
    positive_images_dir = os.path.join(BaseFolder, 'imagesAll_positive/')
    negative_images_dir = os.path.join(BaseFolder, 'sequenceData/negativeOnly/')

    # Move positive images
    move_images(positive_images_dir, destination_dir)

    # Move negative images from each sequence folder
    for seq_num in range(1, 24):  # Assuming sequences are numbered from 1 to 23
        seq_folder = os.path.join(negative_images_dir, f'seq{seq_num}_neg')
        move_images(seq_folder, destination_dir)

    print("DONE.")
