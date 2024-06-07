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
import shutil
from convert_kumc_to_coco import convert


if __name__ == "__main__":

    # Create new folders for formatted images and annotations
    BaseFolder = '/path/to/Kumc22/'
    formatted_output_folder = os.path.join(BaseFolder, "coco/")
    os.makedirs(formatted_output_folder, exist_ok=True)
    formatted_images_folder = os.path.join(formatted_output_folder, "train_images/")
    formatted_ann_folder = os.path.join(formatted_output_folder, "Annotations/")
    os.makedirs(formatted_images_folder, exist_ok=True)
    os.makedirs(formatted_ann_folder, exist_ok=True)

    # Define the paths for images and annotations for training
    trainImagePath = os.path.join(BaseFolder, "PolypsSet/train2019/Image")
    trainAnnotationPath = os.path.join(BaseFolder, "PolypsSet/train2019/Annotation")

    testImagePath = os.path.join(BaseFolder, "PolypsSet/test2019/Image")
    testAnnotationPath = os.path.join(BaseFolder, "PolypsSet/test2019/Annotation")

    valImagePath = os.path.join(BaseFolder, "PolypsSet/val2019/Image")
    valAnnotationPath = os.path.join(BaseFolder, "PolypsSet/val2019/Annotation")

    # loop through subdirectories to rename and move all images to one folder
    print("Moving images and annotations from train...")
    for file in os.listdir(trainImagePath):
        shutil.move(os.path.join(trainImagePath, file), os.path.join(formatted_images_folder, file))

    for file in os.listdir(trainAnnotationPath):
        shutil.move(os.path.join(trainAnnotationPath, file), os.path.join(formatted_ann_folder, file))

    # add the test data
    print("Moving images and annotations from test...")
    for subdir, dirs, files in os.walk(testImagePath):
        if subdir:
            pre = os.path.basename(subdir)
            for file in files:
                newFileName = f"t_{pre}_{file}"
                shutil.move(os.path.join(subdir, file), os.path.join(formatted_images_folder, newFileName))

    for subdir, dirs, files in os.walk(testAnnotationPath):
        if subdir:
            pre = os.path.basename(subdir)
            for file in files:
                newFileName = f"t_{pre}_{file}"
                shutil.move(os.path.join(subdir, file), os.path.join(formatted_ann_folder, newFileName))

    # add the validation data
    print("Moving images and annotations from validation...")
    for subdir, dirs, files in os.walk(valImagePath):
        if subdir:
            pre = os.path.basename(subdir)
            for file in files:
                newFileName = f"v_{pre}_{file}"
                shutil.move(os.path.join(subdir, file), os.path.join(formatted_images_folder, newFileName))

    for subdir, dirs, files in os.walk(valAnnotationPath):
        if subdir:
            pre = os.path.basename(subdir)
            for file in files:
                newFileName = f"v_{pre}_{file}"
                shutil.move(os.path.join(subdir, file), os.path.join(formatted_ann_folder, newFileName))

    print("DONE.")

    # create the annotations
    convert(BaseFolder)
