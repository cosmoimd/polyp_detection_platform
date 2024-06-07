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
import shutil


def convert_to_yolo_format(coco_annotation, img_shape, output_txt_file):
    """
    Convert annotations from COCO format to YOLO format and save them to a specified text file.

    Parameters:
    - coco_annotation (dict): A dictionary containing COCO annotations for a single image.
    - img_shape (tuple of int): A tuple containing the height and width of the image (height, width).
    - output_txt_file (str): The path to the output text file where the YOLO formatted annotations will be saved.

    Each line in the output file represents one object and follows the format:
    <object-class> <x_center> <y_center> <width> <height>
    where coordinates are relative to the width and height of the image, in the range [0, 1].
    """
    with open(output_txt_file, 'w') as f:
        for annotation in coco_annotation['annotations']:
            box = annotation['bbox']
            x_center = ((box[0] + (box[2] / 2)) / img_shape[1])
            y_center = ((box[1] + (box[3] / 2)) / img_shape[0])
            width = box[2] / img_shape[1]
            height = box[3] / img_shape[0]

            # COCO class IDs start at 1, subtract 1 to start at 0 for YOLO
            # class_id = annotation['category_id'] - 1
            class_id = 0

            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def process_images_and_annotations(base_coco_folder, output_train_folder, output_val_folder):
    """
    Process all images and annotations from a COCO dataset, converting the annotations to YOLO format

    Parameters:
    - base_coco_folder (str): The base folder path of the COCO dataset. This folder should contain 'train_ann.json' for
      annotations and a 'train_images' folder for images.
    - output_train_folder (str): The base folder path where the training portion of the converted dataset should be saved.
    - output_val_folder (str): The base folder path where the validation portion of the converted dataset should be saved.

    This function reads COCO annotations, converts them to YOLO format for each image, and copies the images to a new
    directory structure suitable for YOLO-based training.
    """

    with open(os.path.join(base_coco_folder, 'train_ann.json')) as f:
        coco_data = json.load(f)

    image_id_to_file_name = {image['id']: image['file_name'] for image in coco_data['images']}
    image_id_to_shape = {image['id']: (image['height'], image['width']) for image in coco_data['images']}
    annotations_by_image_id = {ann['image_id']: [] for ann in coco_data['annotations']}

    for img, annotation in enumerate(coco_data['annotations']):
        annotations_by_image_id[annotation['image_id']].append(annotation)

    for img, (image_id, img_file_name) in enumerate(image_id_to_file_name.items()):
        print(f"Working on img {img}", end="\r", flush=True)
        img_shape = image_id_to_shape[image_id]

        if img_file_name.startswith("dataset2"):
            curr_out_folder = output_val_folder
        else:
            curr_out_folder = output_train_folder

        output_txt_file = os.path.join(curr_out_folder, 'labels', img_file_name.replace('.jpg', '.txt'))

        if image_id in annotations_by_image_id:
            # Filter annotations for the current image
            current_image_annotations = {
                'annotations': annotations_by_image_id[image_id]
            }
            convert_to_yolo_format(current_image_annotations, img_shape, output_txt_file)
        else:
            # Create an empty file for images without annotations
            open(output_txt_file, 'w').close()

        # Copy image to YOLO folder structure
        source_img_path = os.path.join(base_coco_folder, 'train_images', img_file_name)
        dest_img_path = os.path.join(curr_out_folder, 'images', img_file_name)
        shutil.copy(source_img_path, dest_img_path)


if __name__ == "__main__":
    """ Usage: python split_ucd.py """

    # Parameters; Change paths to desired directories
    base_ucd_folder = "/path/to/UCD/coco/folder"
    output_train_folder = "/path/to/save/UCD/train/split"
    output_val_folder = "/path/to/save/UCD/validation/split"
    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_val_folder, exist_ok=True)

    # Set output folder for YOLO format annotations
    train_images_folder = os.path.join(output_train_folder, "images")
    txt_train_folder = os.path.join(output_train_folder, "labels")
    val_images_folder = os.path.join(output_val_folder, "images")
    txt_val_folder = os.path.join(output_val_folder, "labels")

    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(txt_train_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(txt_val_folder, exist_ok=True)

    print("Creating YOLO format dataset including negative frames...")
    process_images_and_annotations(base_ucd_folder, output_train_folder, output_val_folder)
    print("\nDONE.")
