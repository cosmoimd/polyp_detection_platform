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


"""
    Compute sublist from the test annotations to run experiments on polyp types sublists.
    These sublists include including adenoma versus non-adenoma, diminutive versus non-diminutive,
    and hyperplastic polyps in the sigmoid-rectum, showcasing testing subsets where an optimal AI algorithm should
    demonstrate robust performance.

    Usage:
        - python3 filter_annotations.py <path_to_real_colon_base> <path_to_yolo_dataset_dir>

"""

import os
import shutil
import sys
import json
from glob import glob
import pandas as pd
import xml.etree.ElementTree as ET


def extract_unique_ids(xml_file):
    """
    Extract unique IDs from a single XML file containing annotations.

    Args:
    - xml_file (str): Path to the XML file.

    Returns:
    - list of str: A list containing all unique IDs found within the given XML file.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    unique_ids = []
    for obj in root.findall('.//object'):
        unique_ids.append(obj.find('.//unique_id').text)
    return unique_ids


def track_objects_and_predictions(basepath):
    """
    Track the frames that each object appears in.

    Args:
        - basepath (str): The base path to the dataset containing video annotation folders.

    Returns:
        - dict: keys are unique object IDs and values are lists of frame names in which the object appears.
    """
    object_frames = {}

    # Loop through each video annotation folder in the ground truth
    for annotation_folder in [f for f in sorted(os.listdir(basepath)) if
                              os.path.isdir(os.path.join(basepath, f)) and f.endswith('_annotations')]:
        video_id = annotation_folder.replace('_annotations', '')

        # # If using test set, only select the necessary videos
        # if video_id.split('-')[1] not in ('013', '014', '015'):
        #     continue

        print(f"Processing video {video_id}")
        xml_files = glob(os.path.join(basepath, annotation_folder, '*.xml'))

        # Loop through each annotation xml file
        for xml_file in sorted(xml_files):
            # Format the annotations
            unique_ids = extract_unique_ids(xml_file)
            frame_name = os.path.basename(xml_file).replace('.xml', '')

            # Loop through each annotation in the frame and add the frame
            for unique_id in unique_ids:
                if unique_id in object_frames:
                    object_frames[unique_id].append(frame_name)
                else:
                    object_frames[unique_id] = [frame_name]

    # Sort the lists
    for unique_id in object_frames.keys():
        object_frames[unique_id].sort()

    return object_frames


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python filter_annotations.py <path_to_real_colon_base> <yolo_datasets_dir>")
        sys.exit(1)

    # Specify here dataset base path
    base_dataset_folder = sys.argv[1]
    output_dir = sys.argv[2]

    clinical_data_csv = os.path.join(base_dataset_folder, "lesion_info.csv")
    video_data_csv = os.path.join(base_dataset_folder, "video_info.csv")
    clinical_data = pd.read_csv(clinical_data_csv)
    video_data = pd.read_csv(video_data_csv)

    # Create folders for 1 and 3 seconds
    folder_1s = os.path.join(output_dir, "test_polyps_1s")
    os.makedirs(folder_1s, exist_ok=True)
    folder_3s = os.path.join(output_dir, "test_polyps_3s")
    os.makedirs(folder_3s, exist_ok=True)

    ########################################################
    # Save a testing folders with only positive images within 1s and 3s of polyp appearance
    ########################################################
    polyp_frames = track_objects_and_predictions(base_dataset_folder)

    print("Processing images...")
    oneslist = []

    # Get image ids from the first second of each polyp after finding the fps
    for polyp_id in polyp_frames.keys():
        image_ids = polyp_frames[polyp_id]
        videoname = clinical_data[clinical_data["unique_object_id"] == polyp_id]["unique_video_name"].tolist()[0]
        fps = round(video_data[video_data["unique_video_name"] == videoname]["fps"].tolist()[0])
        oneslist.append(image_ids[:int(1 * fps)])

    src_img_folder = os.path.join(output_dir, "test_real_colon_all/images")
    src_labl_folder = os.path.join(output_dir, "test_real_colon_all/labels")

    img_folder_1s = os.path.join(folder_1s, "images")
    os.makedirs(img_folder_1s, exist_ok=True)
    labl_folder_1s = os.path.join(folder_1s, "labels")
    os.makedirs(labl_folder_1s, exist_ok=True)

    # Initialize coco annotation variables
    coco_anns = os.path.join(output_dir, "test_real_colon_all/test_ann.json")
    with open(coco_anns) as f:
        data = json.load(f)
    all_1s_images = []
    all_3s_images = []

    # Copy images and labels into the new folder for 1s
    for img_ids in oneslist:
        # Copy image file
        for img_id in img_ids:
            source_file = os.path.join(src_img_folder, f'{img_id}.jpg')
            destination_file = os.path.join(img_folder_1s, f'{img_id}.jpg')
            if os.path.exists(source_file):
                if os.path.exists(destination_file) or os.path.islink(destination_file):
                    continue
                os.symlink(source_file, destination_file)
                all_1s_images.append(img_id)
            else:
                print(f"File {source_file} not found.")

            # Copy label file
            source_file = os.path.join(src_labl_folder, f'{img_id}.txt')
            destination_file = os.path.join(labl_folder_1s, f'{img_id}.txt')
            if os.path.exists(source_file):
                if os.path.exists(destination_file) or os.path.islink(destination_file):
                    continue
                shutil.copy(source_file, destination_file)
            else:
                print(f"File {source_file} not found.")

    # Create coco json annotation file
    filtered_annotations = [ann for ann in data['annotations'] if ann['image_id'] in all_1s_images]
    filtered_data = {
        'images': [img for img in data['images'] if img['id'] in all_1s_images],
        'annotations': filtered_annotations,
        'categories': data.get('categories', [])  # Include categories if present
    }

    with open(os.path.join(folder_1s, "test_ann.json"), 'w') as f:
        json.dump(filtered_data, f)

    # Do 3s per polyp
    threeslist = []

    # Get image ids from the first second of each polyp after finding the fps
    for polyp_id in polyp_frames.keys():
        image_ids = polyp_frames[polyp_id]
        videoname = clinical_data[clinical_data["unique_object_id"] == polyp_id]["unique_video_name"].tolist()[0]
        fps = round(video_data[video_data["unique_video_name"] == videoname]["fps"].tolist()[0])
        threeslist.append(image_ids[:int(3 * fps)])

    img_folder_3s = os.path.join(folder_3s, "images")
    os.makedirs(img_folder_3s, exist_ok=True)
    labl_folder_3s = os.path.join(folder_3s, "labels")
    os.makedirs(labl_folder_3s, exist_ok=True)

    # Copy images and labels into the new folder for 1s
    for img_ids in threeslist:
        # Copy image file
        for img_id in img_ids:
            source_file = os.path.join(src_img_folder, f'{img_id}.jpg')
            destination_file = os.path.join(img_folder_3s, f'{img_id}.jpg')
            if os.path.exists(source_file):
                if os.path.exists(destination_file) or os.path.islink(destination_file):
                    continue
                os.symlink(source_file, destination_file)
                all_3s_images.append(img_id)
            else:
                print(f"File {source_file} not found.")

            # Copy label file
            source_file = os.path.join(src_labl_folder, f'{img_id}.txt')
            destination_file = os.path.join(labl_folder_3s, f'{img_id}.txt')
            if os.path.exists(source_file):
                if os.path.exists(destination_file) or os.path.islink(destination_file):
                    continue
                shutil.copy(source_file, destination_file)
            else:
                print(f"File {source_file} not found.")

    # Create coco json annotation file
    filtered_annotations = [ann for ann in data['annotations'] if ann['image_id'] in all_3s_images]
    filtered_data = {
        'images': [img for img in data['images'] if img['id'] in all_3s_images],
        'annotations': filtered_annotations,
        'categories': data.get('categories', [])  # Include categories if present
    }
    with open(os.path.join(folder_3s, "test_ann.json"), 'w') as f:
        json.dump(filtered_data, f)

    print("DONE.")
