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

import cv2
import numpy as np

categoryList = ['polyp']


# writing bbox
def save_bb(txt_path, line):
    with open(txt_path, 'a') as myfile:
        myfile.write(line + "\n")  # append line


def voc_format_v2(class_index, xmin, ymin, xmax, ymax):
    items = map(str, [class_index, xmin, ymin, xmax, ymax])
    return ' '.join(items)


def coco_annotation_format(image_id, category_id, bbox, area, annotation_id, image_width, image_height):
    """
    Formats a single object's annotation into COCO format.

    Args:
        image_id (int): ID of the image the object is in.
        category_id (int): Category ID of the object.
        bbox (list): Bounding box of the object [xmin, ymin, width, height].
        area (float): The area of the bounding box.
        annotation_id (int): Unique ID for the annotation.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        dict: The annotation in COCO format.
    """
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [max(bbox[0], 0),
                 max(0, bbox[1]),
                 min(bbox[2] - max(bbox[0], 0), image_width - max(bbox[0], 0)),
                 min(bbox[3] - max(0, bbox[1]), image_height - max(0, bbox[1]))],
        "area": area,
        "iscrowd": 0
    }


def get_bbox_cordinates_from_mask_coco(image, im_mask, image_id, category_id, annotation_id, image_width, image_height):
    """
    Extracts bounding box coordinates from a mask and formats them for COCO.

    Args:
        image (ndarray): The image corresponding to the mask.
        im_mask (ndarray): Binary mask of the object to extract bounding boxes for.
        image_id (int): ID of the image.
        category_id (int): Category ID of the object.
        annotation_id (int): Starting ID for annotations to be created.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        tuple: A list of annotations in COCO format and the next available annotation ID.
    """

    from skimage.measure import label, regionprops
    label_image = label((im_mask > 0.8).astype(np.uint8))
    properties = regionprops(label_image)
    annotations = []

    for props in properties:
        if props.area > 100:  # You can adjust this threshold
            xmin, ymin, xmax, ymax = props.bbox[1], props.bbox[0], props.bbox[3], props.bbox[2]
            area = props.area
            bbox = [xmin, ymin, xmax, ymax]
            annotation = coco_annotation_format(image_id, 1, bbox, area, annotation_id, image_width, image_height)

            annotations.append(annotation)
            annotation_id += 1

    return annotations, annotation_id
