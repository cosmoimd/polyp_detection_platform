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

""" Use this script to convert the annotation of the REAL-Colon dataset from the VOC format to the Yolo format.
    The script allows to include in the converted dataset a subset of the whole dataset, selecting the number of positive and negative images.
    The script will also produce 3 splits (training, validation, testing), with same proportion across each dataset group (1-4)

    Copyright 2023-, Cosmo Intelligent Medical Devices
"""

import os
import json
import random
import xml.etree.ElementTree as ET


def parsevocfile(annotation_file):
    """ Parse an annotation file in voc format

        Example VOC notation:
            <annotation>
                </version_fmt>1.0<version_fmt>
                <folder>002-001_frames</folder>
                <filename>002-001_18185.jpg</filename>
                <source>
                    <database>cosmoimd</database>
                    <release>v1.0_20230228</release>
                </source>
                <size>
                    <width>1240</width>
                    <height>1080</height>
                    <depth>3</depth>
                </size>
                <object>
                    <name>lesion</name>
                    <unique_id>videoname_lesionid</unique_id>
                    <box_id>1</box_id>  <- id of the box within the image
                    <bndbox>
                        <xmin>540</xmin>
                        <xmax>1196</xmax>
                        <ymin>852</ymin>
                        <ymax>1070</ymax>
                    </bndbox>
                </object>
            </annotation>""

    Args:
        annotation_file (str): Path to the VOC format XML annotation file.

    Returns:
        dict: A dictionary containing two keys:
              - 'boxes': a list of dictionaries, each representing one bounding box with keys 'name', 'unique_id', 'box_id', and 'box_ltrb' (list of xmin, ymin, xmax, ymax).
              - 'img_shape': a tuple representing the image shape (height, width, depth).
              - 'img_name': the filename of the image.
    """

    if not os.path.exists(annotation_file):
        raise Exception("Cannot find bounding box file %s" % annotation_file)
    try:
        tree = ET.parse(annotation_file)
    except Exception as e:
        print(e)
        raise Exception("Failed to open annotation file %s" % annotation_file)

    # Read all the boxes
    img = {}
    cboxes = []
    for elem in tree.iter():
        # Get the image full path from the image name and folder, not from the annotation tag
        if 'filename' in elem.tag:
            filename = elem.text
        if 'width' in elem.tag:
            img['width'] = int(elem.text)
        if 'height' in elem.tag:
            img['height'] = int(elem.text)
        if 'depth' in elem.tag:
            img['depth'] = int(elem.text)
        if 'object' in elem.tag or 'part' in elem.tag:
            obj = {}

            # create empty dict where store properties
            for attr in list(elem):
                if 'name' in attr.tag:
                    obj['name'] = attr.text
                if 'unique_id' in attr.tag:
                    obj['unique_id'] = attr.text

                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            l = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            t = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            r = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            b = int(round(float(dim.text)))

                    obj["box_ltrb"] = [l, t, r, b]
            cboxes.append(obj)
    img_shape = (img["height"], img["width"], img["depth"])
    return {"boxes": cboxes, "img_shape": img_shape, "img_name": filename}


def convert_to_yolo_format(c_data, output_txt_file):
    """
    Converts annotations from the coco format to yolo formatting

    Args:
        c_data (dict): dictionary of bounding box annotations in coco format
        output_txt_file (str): location of the corresponding yolo txt file to save the annotations to

    Return:
        None. Outputs annotations to the txt file.
    """
    # Write the bounding box annotations to the .txt file
    with open(output_txt_file, 'w') as f:
        for cbox in c_data['boxes']:
            x_center = ((cbox['box_ltrb'][2] + cbox['box_ltrb'][0]) / 2) / c_data['img_shape'][1]
            y_center = ((cbox['box_ltrb'][3] + cbox['box_ltrb'][1]) / 2) / c_data['img_shape'][0]
            width = (cbox['box_ltrb'][2] - cbox['box_ltrb'][0]) / c_data['img_shape'][1]
            height = (cbox['box_ltrb'][3] - cbox['box_ltrb'][1]) / c_data['img_shape'][0]
            class_id = 0  # Assuming you have only one class 'lesion'. Change as per your requirement.
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def convert_video_list(base_dataset_folder, video_list, annotation_list, frames_output_folder, txt_output_folder,
                       json_ann_file, negative_ratio=0, num_positives_per_lesions=-1):
    """
    Takes in input a list of video folders (each of them contains the video frames) and the relative annotation folders and
    convert them into Yolo format. All frames with boxes are added to the dataset, while the negative frames are randomly selected
    from the whole dataset. We select N negative frames where N = max(1% of #negative_frames, 10% of #frames_with_boxes)

    Args:
        base_dataset_folder (str): Base folder for the REAL-Colon dataset in the original format.
        video_list (list of str): List of video folders for conversion.
        annotation_list (list of str): List of annotation folders corresponding to the video folders.
        frames_output_folder (str): Output folder for the frames (symbolic links will be created here).
        txt_output_folder (str): Output folder for the YOLO formatted annotation text files.
        json_ann_file (str): Path to the output JSON file with annotations in COCO format.
        negative_ratio (float): Ratio of frames without boxes to include, relative to the number of frames with boxes [0, 1].
        num_positives_per_lesions (int): Number of positive frames to keep for each lesion. Use -1 to keep all.
    """

    # Check input parameters are valid
    if negative_ratio < 0 or negative_ratio > 1:
        raise Exception(f"Invalid 'negative_ratio' arg {negative_ratio}, must be in [0,1]")

    # create output folder
    os.makedirs(frames_output_folder, exist_ok=True)

    # Hardcoded dictionary fields
    data = {}
    data['info'] = {'description': 'Cosmo data', 'url': 'http://cosmoimd.com', 'version': '1.0', 'year': 2023,
                    'contributor': 'CosmoIMD', 'date_created': '2023/02/28'}
    data['licenses'] = [{'url': 'https://creativecommons.org/licenses/by-nc-sa/4.0/', 'id': 1,
                         'name': 'Attribution-NonCommercial-ShareAlike License'}]
    data['categories'] = [{'supercategory': 'lesion', 'id': 0, 'name': 'lesion'}]
    data['images'] = []
    data['annotations'] = []

    # Process each video: subsample frames and convert
    image_uniq_id_cnt = 0
    image_uniq_box_cnt = 0
    uniq_box_to_lesion_association = {}
    for video_idx, (curr_video_folder, curr_ann_folder) in enumerate(zip(video_list, annotation_list)):
        print(f"Processing video {video_idx}")
        all_images = sorted(os.listdir(os.path.join(base_dataset_folder, curr_video_folder)),
                            key=lambda x: int(x.split("_")[-1].split(".")[0]))
        all_xmls = sorted(os.listdir(os.path.join(base_dataset_folder, curr_ann_folder)),
                          key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if not len(all_images) == len(all_xmls):
            raise Exception("Image and annotations must have same length")

        # Only select a subsets of XMLS that are useful for training
        all_datas = []
        num_boxes_indexes = []
        for c_xml in all_xmls:
            c_data = parsevocfile(os.path.join(base_dataset_folder, curr_ann_folder, c_xml))
            all_datas.append(c_data)
            num_boxes_indexes.append(len(c_data['boxes']))

        # prepare a dictionary with the list of frames for each lesion
        frames_wbox_indexes = [idx for idx, v in enumerate(num_boxes_indexes) if v > 0]
        frames_nobox_indexes = [idx for idx, v in enumerate(num_boxes_indexes) if v == 0]
        per_lesion_dict = {}
        for cidx, c_data in enumerate(all_datas):
            for cbox in c_data['boxes']:
                cname = cbox['unique_id']
                if not cname in per_lesion_dict.keys():
                    per_lesion_dict[cname] = []
                per_lesion_dict[cname].append(cidx)
        print(
            f"Found {len(per_lesion_dict)} lesions with {' - '.join([str(len(per_lesion_dict[x])) for x in per_lesion_dict.keys()])} frames each")

        # Select the positive samples
        random.seed(1000)
        selected_frames_w_box_indexes = set([])
        for l in per_lesion_dict.keys():
            c_list = per_lesion_dict[l]
            if num_positives_per_lesions > 0:
                random.shuffle(c_list)
                to_select = min(len(c_list), num_positives_per_lesions)
                selected_frames_w_box_indexes = selected_frames_w_box_indexes.union(set(c_list[:to_select]))
            else:
                selected_frames_w_box_indexes = selected_frames_w_box_indexes.union(set(c_list))
        selected_frames_w_box_indexes = sorted(list(selected_frames_w_box_indexes))
        print(
            f"Sampled {num_positives_per_lesions} positive frames per lesion, using {len(selected_frames_w_box_indexes)}/{len(frames_wbox_indexes)} positive frames")

        # Select the negative samples
        to_keep = int(negative_ratio * len(frames_nobox_indexes))
        selected_frames = selected_frames_w_box_indexes + random.sample(frames_nobox_indexes, to_keep)
        print(
            f"Sampled {to_keep} negative frames from frames {len(frames_nobox_indexes)} total negatives (negative_ratio = {negative_ratio})")
        xml_to_be_used = [all_xmls[y] for y in selected_frames]

        for c_xml in xml_to_be_used:
            c_data = parsevocfile(os.path.join(base_dataset_folder, curr_ann_folder, c_xml))
            output_txt_file = os.path.join(txt_output_folder, os.path.splitext(c_data['img_name'])[0] + '.txt')
            convert_to_yolo_format(c_data, output_txt_file)

            # Create symbolic link from the original dataset location
            os.symlink(
                os.path.join(base_dataset_folder, curr_video_folder, c_data['img_name']),
                os.path.join(frames_output_folder, c_data['img_name']))

            # Add the image to the list of images
            data["images"].append({'license': 1, 'file_name': c_data['img_name'], 'height': c_data['img_shape'][0],
                                   'width': c_data['img_shape'][1], 'id': c_data['img_name'].split(".")[0]})

            # Loop on boxes
            for cbox in c_data['boxes']:
                l = min(cbox['box_ltrb'][0], c_data['img_shape'][1] - 1)
                t = min(cbox['box_ltrb'][1], c_data['img_shape'][0] - 1)
                r = min(cbox['box_ltrb'][2], c_data['img_shape'][1] - 1)
                b = min(cbox['box_ltrb'][3], c_data['img_shape'][0] - 1)
                area = (b - t) * (r - l)
                data['annotations'].append({'segmentation': [[l, t, r, t, r, b, l, b]], 'area': area,
                                            'iscrowd': 0, 'image_id': c_data['img_name'].split(".")[0],
                                            'unique_id': cbox['unique_id'],
                                            'bbox': [l, t, r - l, b - t], 'category_id': 0, 'id': image_uniq_box_cnt})
                if not cbox['unique_id'] in uniq_box_to_lesion_association.keys():
                    uniq_box_to_lesion_association[cbox['unique_id']] = []
                uniq_box_to_lesion_association[cbox['unique_id']].append(image_uniq_box_cnt)
                image_uniq_box_cnt += 1

            # Image process completed, increment id
            image_uniq_id_cnt += 1

    print(f"Processing completed with {image_uniq_id_cnt} images and {image_uniq_box_cnt} boxes")
    with open(json_ann_file, 'w') as off:
        json.dump(data, off)

if __name__ == "__main__":
    # Parameters
    base_dataset_folder = "/path/to/real/colon/base"  # Path to the folder of the original REAL-COLON dataset (update with proper value)
    output_folder = f"/output/folder"  # Output folder for the converted dataset
    num_positives_per_lesions = 1000000  # Number of frames with boxes for each polyp to be included in the output dataset
    negative_ratio = 1  # Ratio of images without boxes for each video to be included in the output dataset [0,1]
    NUM_TRAIN_VIDEOS_PER_SET = 0  # the first videos for each set will go in the train set
    NUM_VALID_VIDEOS_PER_SET = 0  # the next in the validation set, and the remaining videos for each set will go in the test set
    # read input data
    video_list = sorted([x for x in os.listdir(base_dataset_folder) if x.endswith("_frames")])
    annotation_list = sorted([x for x in os.listdir(base_dataset_folder) if x.endswith("_annotations")])

    # # Further filter to keep only studies that start with "001" if needed
    # video_list = [x for x in video_list if x.startswith("004")]
    # annotation_list = [x for x in annotation_list if x.startswith("004")]

    # create folders for train, val, and test
    os.makedirs(output_folder, exist_ok=False)
    train_folder = os.path.join(output_folder, "train")
    train_ann_json = os.path.join(train_folder, "train_ann.json")
    val_folder = os.path.join(output_folder, "val")
    val_ann_json = os.path.join(val_folder, "val_ann.json")
    test_folder = os.path.join(output_folder, "test")
    test_ann_json = os.path.join(test_folder, "test_ann.json")
    os.makedirs(train_folder, exist_ok=False)
    os.makedirs(val_folder, exist_ok=False)
    os.makedirs(test_folder, exist_ok=False)

    # set output folder for yolo format annotations
    train_images_folder = os.path.join(train_folder, "images")
    validation_images_folder = os.path.join(val_folder, "images")
    test_images_folder = os.path.join(test_folder, "images")
    txt_train_folder = os.path.join(train_folder, "labels")
    txt_val_folder = os.path.join(val_folder, "labels")
    txt_test_folder = os.path.join(test_folder, "labels")
    os.makedirs(txt_train_folder, exist_ok=False)
    os.makedirs(txt_val_folder, exist_ok=False)
    os.makedirs(txt_test_folder, exist_ok=False)

    # Perform the conversion:
    video_list_train = [x for x in video_list if int(x.split("-")[1].split("_")[0]) <= NUM_TRAIN_VIDEOS_PER_SET]
    annotation_list_train = [x for x in annotation_list if
                             int(x.split("-")[1].split("_")[0]) <= NUM_TRAIN_VIDEOS_PER_SET]
    convert_video_list(base_dataset_folder, video_list_train, annotation_list_train, train_images_folder,
                       txt_train_folder, train_ann_json, negative_ratio=negative_ratio,
                       num_positives_per_lesions=num_positives_per_lesions)
    print("Training subset conversion completed")
    video_list_validation = [x for x in video_list if (
                int(x.split("-")[1].split("_")[0]) > NUM_TRAIN_VIDEOS_PER_SET and int(
            x.split("-")[1].split("_")[0]) <= (NUM_VALID_VIDEOS_PER_SET + NUM_TRAIN_VIDEOS_PER_SET))]
    annotation_list_validation = [x for x in annotation_list if (
                int(x.split("-")[1].split("_")[0]) > NUM_TRAIN_VIDEOS_PER_SET and int(
            x.split("-")[1].split("_")[0]) <= (NUM_VALID_VIDEOS_PER_SET + NUM_TRAIN_VIDEOS_PER_SET))]
    convert_video_list(base_dataset_folder, video_list_validation, annotation_list_validation, validation_images_folder,
                       txt_val_folder, val_ann_json, negative_ratio=negative_ratio,
                       num_positives_per_lesions=num_positives_per_lesions)
    print("Validation subset conversion completed")
    video_list_test = [x for x in video_list if
                       int(x.split("-")[1].split("_")[0]) > (NUM_VALID_VIDEOS_PER_SET + NUM_TRAIN_VIDEOS_PER_SET)]
    annotation_list_test = [x for x in annotation_list if
                            int(x.split("-")[1].split("_")[0]) > (NUM_VALID_VIDEOS_PER_SET + NUM_TRAIN_VIDEOS_PER_SET)]
    convert_video_list(base_dataset_folder, video_list_test, annotation_list_test, test_images_folder,
                       txt_test_folder, test_ann_json, negative_ratio=negative_ratio,
                       num_positives_per_lesions=num_positives_per_lesions)
    print("Testing subset conversion completed")
