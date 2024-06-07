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

import xml.etree.ElementTree as ET
from glob import glob
import sys
import os


def frame_parser(basepath, prediction_dir, test_on):
    """
    Parses frame data from XML annotations and prediction files, categorizing frames based on
    the presence of polyps, their detections, and the predictions made by a model.

    Parameters:
    - basepath (str): The base path to the directory containing the video annotation folders.
    - prediction_dir (str): The directory containing prediction files for each frame.
    - test_on (str): Specifies the subset of data to be used. It can be a specific study
      ('study_00<study_num>'), 'all' for all data, or 'test_set' for a predefined set of test videos.

    Returns:
    - tuple: Contains three dictionaries:
        - all_polyp_frames: Maps each unique polyp identifier to a list of frame numbers where the polyp appears.
        - all_detected_frames: Maps each unique polyp identifier to a dictionary of frame numbers and their
          corresponding detection confidence and Intersection over Union (IoU) values.
        - all_pred: Maps frame names to a list of detection confidence and IoU values for predictions made by the model.
    """
    # Initialize dictionaries for tracking frames
    all_polyp_frames = {}
    all_detected_frames = {}
    all_pred = {}

    # Loop through each video annotation folder in the ground truth
    for annotation_folder in [f for f in sorted(os.listdir(basepath)) if
                              os.path.isdir(os.path.join(basepath, f)) and f.endswith('_annotations')]:
        video_id = annotation_folder.replace('_annotations', '')

        # Check what is being tested and adjust accordingly
        if test_on.startswith("study"):
            curr_study = [test_on.split("_")[1]]
            videos_list = ['006', '007', '008', '009', '010', '011', '012', '013', '014', '015']
            if curr_study[0] not in ["001", "002", "003", "004"]:
                print("Invalid choice. test_on param must be study_00<study_num>, all, or test_set.")
                sys.exit("Error: The choice was not one of the allowed options.")
        elif test_on == "all":
            curr_study = ["001", "002", "003", "004"]
            videos_list = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012',
                           '013', '014', '015']
        elif test_on == "all_min_train" or "test_set":
            curr_study = ["001", "002", "003", "004"]
            videos_list = ['006', '007', '008', '009', '010', '011', '012', '013', '014', '015']
        else:
            print("Invalid choice. test_on param must be study_00<study_num>, all, or test_set.")
            sys.exit("Error: The choice was not one of the allowed options.")

        # If using test set, only select the necessary videos
        if video_id.split('-')[1] not in videos_list or video_id.split('-')[0] not in curr_study:
            continue

        print(f"Processing video {video_id}...")
        xml_files = glob(os.path.join(basepath, annotation_folder, '*.xml'))

        # Loop through each annotation xml file
        sorted_xmls = sorted(xml_files,
                             key=lambda x: (int(float(os.path.basename(x).replace('.xml', '').split('_')[1]))))
        for xml_file in sorted_xmls:
            # Format the annotations
            annotations = extract_annotations(xml_file)
            frame_name = os.path.basename(xml_file).replace('.xml', '')
            frame_number = int(float(frame_name.split('_')[-1]))

            all_pred[frame_name] = []

            # Loop through each annotation in the frame and add the intro frame if it is a new lesion
            for annotation in annotations:
                unique_id = annotation['unique_id']
                if unique_id not in all_polyp_frames.keys():
                    all_polyp_frames[unique_id] = [frame_number]
                else:
                    all_polyp_frames[unique_id].append(frame_number)

            # Get the prediction file for that frame and extract the predictions
            pred_file_path = os.path.join(prediction_dir, f"{frame_name}.txt")

            if os.path.exists(pred_file_path):
                with open(pred_file_path, 'r') as f:
                    predictions = f.readlines()

                for line in predictions:
                    content = line.strip().split()
                    bbox_pred = [float(i) for i in content[1:5]]
                    confidence = float(content[5])

                    if not annotations:
                        all_pred[frame_name].append([confidence, -1])

                    # Check if the annotation matches, and if it does record the detection for the polyp
                    for annotation in annotations:
                        bbox_gt = [annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax']]
                        iou = compute_iou(bbox_pred, bbox_gt, annotation['img_width'], annotation['img_height'])

                        unique_id = annotation['unique_id']

                        if unique_id not in all_detected_frames.keys():
                            all_detected_frames[unique_id] = {frame_number: [confidence, iou]}
                        else:
                            if frame_number not in all_detected_frames[unique_id].keys():
                                all_detected_frames[unique_id][frame_number] = [confidence, iou]
                        all_pred[frame_name].append([confidence, iou])
            else:
                if annotations:
                    all_pred[frame_name].append([0, 0])

    for unique_id in all_polyp_frames:
        all_polyp_frames[unique_id] = sorted(all_polyp_frames[unique_id])

    sorted_all_pred = {key: all_pred[key] for key in
                       sorted(all_pred, key=lambda x: (x.split('_')[0], int(float(x.split('_')[1]))))}

    return all_polyp_frames, all_detected_frames, sorted_all_pred


def compute_iou(box1, box2, image_width, image_height):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
        - box1 (list of float): The first bounding box in normalized coordinates [center_x, center_y, width, height].
        - box2 (list of float): The second bounding box in absolute coordinates [xmin, ymin, xmax, ymax].
        - image_width (int): The width of the image to scale the normalized coordinates of box1.
        - image_height (int): The height of the image to scale the normalized coordinates of box1.

    Returns:
        - float: The IoU value.
    """
    # Convert box1 from normalized coords to absolute pixel coords
    center_x, center_y, width, height = box1
    x1_pred = (center_x - width / 2) * image_width
    x2_pred = (center_x + width / 2) * image_width
    y1_pred = (center_y - height / 2) * image_height
    y2_pred = (center_y + height / 2) * image_height

    # box2 is already in absolute coordinates
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    # Calculate intersection
    xi1 = max(x1_pred, x1_gt)
    yi1 = max(y1_pred, y1_gt)
    xi2 = min(x2_pred, x2_gt)
    yi2 = min(y2_pred, y2_gt)
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate union
    box1_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union = box1_area + box2_area - intersection

    # Calculate IoU
    iou = intersection / union if union != 0 else 0

    return iou


def extract_annotations(xml_file):
    """
    Extract annotations from an XML file.

    Parameters:
        - xml_file (str): Path to the XML file containing annotations.

    Returns:
        - list of dict: A list where each dictionary contains details of one bounding box and associated metadata.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []
    for obj in root.findall('.//object'):
        box_data = {
            'filename': root.find('.//filename').text,
            'unique_id': obj.find('.//unique_id').text,
            'box_id': obj.find('.//box_id').text,
            'img_width': int(root.find('.//size/width').text),
            'img_height': int(root.find('.//size/height').text),
            'xmin': float(obj.find('.//bndbox/xmin').text),
            'xmax': float(obj.find('.//bndbox/xmax').text),
            'ymin': float(obj.find('.//bndbox/ymin').text),
            'ymax': float(obj.find('.//bndbox/ymax').text)
        }
        annotations.append(box_data)
    return annotations
