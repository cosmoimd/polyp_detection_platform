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
import pickle
import os
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


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
        elif test_on == "test_set":
            curr_study = ["001", "002", "003", "004"]
            videos_list = ['006', '007', '008', '009', '010', '011', '012', '013', '014', '015']
        elif test_on == "debug":
            curr_study = ["001"]
            videos_list = ['006']
        else:
            print("Invalid choice. test_on param must be study_00<study_num>, all, or test_set.")
            sys.exit("Error: The choice was not one of the allowed options.")

        # If using test set, only select the necessary videos
        if video_id.split('-')[1] not in videos_list or video_id.split('-')[0] not in curr_study:
            continue

        print(f"Processing video {video_id}...")
        xml_files = glob(os.path.join(basepath, annotation_folder, '*.xml'))

        # Loop through each annotation xml file
        for xml_file in sorted(xml_files):
            # Format the annotations
            annotations = extract_annotations(xml_file)
            frame_name = os.path.basename(xml_file).replace('.xml', '')

            if annotations:
                all_pred[frame_name] = [1]

            else:
                all_pred[frame_name] = [0]

            # Get the prediction file for that frame and extract the predictionsw
            pred_file_path = os.path.join(prediction_dir, f"{frame_name}.txt")

            if os.path.exists(pred_file_path):
                with open(pred_file_path, 'r') as f:
                    predictions = f.readlines()

                for line in predictions:
                    content = line.strip().split()
                    bbox_pred = [float(i) for i in content[1:5]]
                    confidence = float(content[5])

                    if not annotations:
                        all_pred[frame_name].append([confidence, 0])

                    # Check if the annotation matches, and if it does record the detection for the polyp
                    for annotation in annotations:
                        bbox_gt = [annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax']]
                        iou = compute_iou(bbox_pred, bbox_gt, annotation['img_width'], annotation['img_height'])

                        all_pred[frame_name].append([confidence, iou])

    sorted_all_pred = {key: all_pred[key] for key in
                       sorted(all_pred, key=lambda x: (x.split('_')[0], int(float(x.split('_')[1]))))}

    return sorted_all_pred


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


def bootstrap_roc(true_labels, predicted_scores, n_bootstrap=1):
    bootstrapped_tpr = []
    bootstrapped_fpr = []
    # Create dynamic thresholds based on quantiles and linear spacing
    quantile_values = np.quantile(predicted_scores, np.linspace(0, 1, 100))
    linear_values = np.linspace(predicted_scores.min(), predicted_scores.max(), 400)
    thresholds = np.unique(np.concatenate([quantile_values, linear_values]))  # Remove duplicates

    for i in range(n_bootstrap):
        print("Bootstrap N ", i + 1)
        # Bootstrap sampling
        indices = resample(np.arange(len(true_labels)))
        bs_labels = true_labels[indices]
        bs_scores = predicted_scores[indices]

        # Pre-calculate positives and negatives
        positives = bs_labels == 1
        negatives = bs_labels == 0
        pos_count = positives.sum()
        neg_count = negatives.sum()

        tprs = []
        fprs = []
        for threshold in thresholds:
            # Calculate TPR and FPR at each threshold using pre-calculated sums
            tpr = np.sum((bs_scores >= threshold) & positives) / pos_count
            fpr = np.sum((bs_scores >= threshold) & negatives) / neg_count
            tprs.append(tpr)
            fprs.append(fpr)

        bootstrapped_tpr.append(tprs)
        bootstrapped_fpr.append(fprs)

    # Calculate mean and 95% CI for TPR and FPR at each threshold
    mean_tpr = np.mean(bootstrapped_tpr, axis=0)
    lower_tpr = np.percentile(bootstrapped_tpr, 2.5, axis=0)
    upper_tpr = np.percentile(bootstrapped_tpr, 97.5, axis=0)
    mean_fpr = np.mean(bootstrapped_fpr, axis=0)

    return mean_fpr, mean_tpr, lower_tpr, upper_tpr, thresholds


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python standard_ROC.py <real_colon_base_path> <pred_labels_path> <test_on> <output_path>")
        sys.exit(1)
    basepath = sys.argv[1]
    pred_labels = sys.argv[2]
    test_on = sys.argv[3]
    output_path = sys.argv[4]
    os.makedirs(output_path, exist_ok=True)

    pred_dict = frame_parser(basepath, pred_labels, test_on)

    # Extract true labels and predicted scores per frame
    true_labels = []
    predicted_scores = []
    for frame_id, data in pred_dict.items():
        frame_label = data[0]  # Fixed missing frame_label extraction
        if len(data) == 1:
            max_det_score = 0
        else:
            detections = data[1:]
            if frame_label == 1:
                # Consider only Iou greater than 0.2, take them max
                max_det_score = max([detection[0] for detection in detections if detection[1] >= 0.2], default=0)
            else:
                max_det_score = max([detection[0] for detection in detections], default=0)
        true_labels.append(frame_label)
        predicted_scores.append(max_det_score)
    true_labels = np.array(true_labels)
    predicted_scores = np.array(predicted_scores)

    # Save true_labels and predicted_scores to a pickle file
    with open(os.path.join(output_path, 'labels_scores.pkl'), 'wb') as f:
        pickle.dump((true_labels, predicted_scores), f)

    print(f"Saved true_labels and predicted_scores to {os.path.join(output_path, 'labels_scores.pkl')}")

    pickle_file_path = os.path.join(output_path, 'labels_scores.pkl')
    # Load the true_labels and predicted_scores from the pickle file
    with open(pickle_file_path, 'rb') as f:
        true_labels, predicted_scores = pickle.load(f)

    # Perform bootstrapping
    mean_fpr, mean_tpr, lower_tpr, upper_tpr, thresholds = bootstrap_roc(true_labels, predicted_scores)
    # Compute and print the AUC
    auc_score = roc_auc_score(true_labels, predicted_scores)

    label = 'UCD'
    color = 'blue'
    # Color for each model

    # Save the ROC data to a pickle file
    roc_data = {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'lower_tpr': lower_tpr,
        'upper_tpr': upper_tpr,
        'thresholds': thresholds,
        'auc_score': auc_score
    }
    save_path = os.path.join(output_path, f'{label}_roc_data.pkl')
    with open(save_path, 'wb') as handle:
        pickle.dump(roc_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Construct the path to the pickle file
    load_path = os.path.join(output_path, f'{label}_roc_data.pkl')

    # Open the pickle file and load the data
    with open(load_path, 'rb') as handle:
        roc_data = pickle.load(handle)

    # Now you can access the data from the loaded roc_data dictionary
    mean_fpr = roc_data['mean_fpr']
    mean_tpr = roc_data['mean_tpr']
    lower_tpr = roc_data['lower_tpr']
    upper_tpr = roc_data['upper_tpr']
    thresholds = roc_data['thresholds']
    auc_score = roc_data['auc_score']

    # # Plot the ROC curve for this model
    # plt.fill_between(mean_fpr, lower_tpr, upper_tpr, color=color, alpha=0.5)

    # Plot the ROC curve
    plt.figure(figsize=(10, 7))
    plt.plot(mean_fpr[1:], mean_tpr[1:], label=f'{label}, AUC: {auc_score:.3f}', color=color)

    # Customize the plot
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.title('ROC on the whole RC', fontsize=22)
    plt.legend(title="Model", loc='lower right', fontsize=16, title_fontsize=18)

    # Set limits for x and y axes
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(True)

    # Save the figure
    plt.savefig(os.path.join(output_path, 'roc_curve_with_bootstrapping.png'), dpi=300)
