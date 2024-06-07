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


import os
import json

def create_symbolic_links(coco_folders, new_images_folder):
    """
    Creates symbolic links for images from multiple COCO datasets in a new directory.

    Args:
        coco_folders (list of str): List of paths to COCO dataset folders.
        new_images_folder (str): Path to the directory where symbolic links will be created.

    Returns:
        dict: A dictionary mapping the unique image names to their symbolic link paths.
    """

    created_links = {}
    dataset_id = 0  # Initialize a dataset identifier

    # Loop through each image in every folder
    for folder in coco_folders:
        img_folder = os.path.join(folder, "train_images")
        for image in os.listdir(img_folder):
            # Construct a unique filename by appending the dataset_id to the original filename
            unique_image_name = f"dataset{dataset_id}_{image}"

            # Create the symbolic link
            if unique_image_name not in created_links:
                link_path = os.path.join(new_images_folder, unique_image_name)
                target_path = os.path.join(img_folder, image)
                if not os.path.exists(link_path):
                    os.symlink(target_path, link_path)
                created_links[unique_image_name] = link_path
        dataset_id += 1  # Increment the dataset_id for the next dataset

    print("Symbolic links created successfully.")
    return created_links


def combine_annotations(coco_folders, new_anns_file, created_links):
    """
    Combines annotations from multiple COCO datasets into a single JSON file

    Args:
        coco_folders (list of str): List of paths to COCO dataset folders containing annotations.
        new_anns_file (str): Path to the new combined annotations JSON file.
        created_links (dict): Dictionary of unique image names to their symbolic link paths

    Returns:
        None. The combined annotations are saved to a new JSON file.
    """

    combined_images, combined_annotations = [], []
    image_id_mapping, annotation_id = {}, 1
    dataset_id = 0  # Initialize a dataset identifier

    # Loop through each annotation file
    for folder in coco_folders:
        ann_file = os.path.join(folder, 'train_ann.json')
        with open(ann_file, 'r') as file:
            data = json.load(file)

            for img in data['images']:
                # Use the unique image name
                unique_image_name = f"dataset{dataset_id}_{img['file_name']}"

                # update id and file name to the new symlink one
                if unique_image_name in created_links:
                    new_image_id = len(combined_images) + 1
                    image_id_mapping[img['id']] = new_image_id
                    img['id'] = new_image_id
                    img['file_name'] = os.path.basename(created_links[unique_image_name])
                    combined_images.append(img)

            # Add the annotation
            for ann in data['annotations']:
                if ann['image_id'] in image_id_mapping:
                    ann['id'] = annotation_id
                    ann['image_id'] = image_id_mapping[ann['image_id']]
                    combined_annotations.append(ann)
                    annotation_id += 1

        dataset_id += 1  # Increment the dataset_id for the next dataset

    # Save combined dataset
    combined_coco_dataset = {
        "images": combined_images,
        "annotations": combined_annotations,
        "categories": data['categories']  # Assuming categories are consistent across datasets
    }

    with open(new_anns_file, 'w') as f:
        json.dump(combined_coco_dataset, f)

    print("Combined annotations json created successfully.")


if __name__ == "__main__":
    # Define paths to the coco datasets and new paths for the combined
    coco_folders = ['/path/to/LDPolypVideo/coco', '/path/to/Kumc22/coco',
                    '/path/to/PolypGen2021_MultiCenterData_v3/coco',
                    '/path/to/sun_database/coco']
    new_output_folder = '/path/to/output/folder'
    os.makedirs(new_output_folder, exist_ok=True)
    new_images_folder = os.path.join(new_output_folder, 'train_images')
    new_anns_file = '/path/to/combined_coco/train_ann.json'

    if not os.path.exists(new_images_folder):
        os.makedirs(new_images_folder)

    print("Creating Symbolic Links...")
    created_links = create_symbolic_links(coco_folders, new_images_folder)

    print("Creating combined annotation json...")
    combine_annotations(coco_folders, new_anns_file, created_links)

    print("DONE.")
