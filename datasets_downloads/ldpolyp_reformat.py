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
from convert_ldpolyp_to_coco import convert


if __name__ == "__main__":

    # Create new folders for formatted images and annotations
    BaseFolder = '/path/to/LDPolypVideo/'
    formatted_output_folder = os.path.join(BaseFolder, "coco/")
    os.makedirs(formatted_output_folder, exist_ok=True)
    formatted_images_folder = os.path.join(formatted_output_folder, "train_images/")
    formatted_ann_folder = os.path.join(formatted_output_folder, "Annotations/")
    os.makedirs(formatted_images_folder, exist_ok=True)
    os.makedirs(formatted_ann_folder, exist_ok=True)

    # Define the paths for images and annotations for training
    trainImagePath = os.path.join(BaseFolder, "LDPolypVideo-20240108T135538Z-002/LDPolypVideo/TrainValid/Images")
    trainAnnotationPath = os.path.join(BaseFolder,
                                       "LDPolypVideo-20240108T135538Z-002/LDPolypVideo/TrainValid/Annotations")

    testImagePath = os.path.join(BaseFolder, "LDPolypVideo-20240108T135538Z-003/LDPolypVideo/Test/Images")
    testAnnotationPath = os.path.join(BaseFolder, "LDPolypVideo-20240108T135538Z-003/LDPolypVideo/Test/Annotations")

    negImagePath = os.path.join(BaseFolder, "extracted_negative_frames")

    # loop through subdirectories to rename and move all images to one folder
    print("Moving images...")
    for subdir, dirs, files in os.walk(trainImagePath):
        if subdir:
            pre = os.path.basename(subdir)
            for file in files:
                newFileName = f"{pre}_{file}"
                shutil.move(os.path.join(subdir, file), os.path.join(formatted_images_folder, newFileName))

    for subdir, dirs, files in os.walk(testImagePath):
        if subdir:
            pre = os.path.basename(subdir)
            for file in files:
                newFileName = f"{pre}_{file}"
                shutil.move(os.path.join(subdir, file), os.path.join(formatted_images_folder, newFileName))

    print("Moving and creating annotations for extracted negative frames...")
    for file in os.listdir(negImagePath):
        # Check if it's a file and not a directory
        if os.path.isfile(os.path.join(negImagePath, file)):
            # Construct the new filename and annotation filename
            newFileName = f"negative_{file}"
            annotationFileName = f"negative_{os.path.splitext(file)[0]}.txt"

            # Move the image file
            shutil.move(os.path.join(negImagePath, file), os.path.join(formatted_images_folder, newFileName))

            # Create an annotation file with "0"
            with open(os.path.join(formatted_ann_folder, annotationFileName), 'w') as f:
                f.write("0")

    # loop through subdirectories to rename and move all annotations to one folder
    print("Moving annotations...")
    for subdir, dirs, files in os.walk(trainAnnotationPath):
        if subdir:
            pre = os.path.basename(subdir)
            for file in files:
                newFileName = f"{pre}_{file}"
                shutil.move(os.path.join(subdir, file), os.path.join(formatted_ann_folder, newFileName))

    for subdir, dirs, files in os.walk(testAnnotationPath):
        if subdir:
            pre = os.path.basename(subdir)
            for file in files:
                newFileName = f"{pre}_{file}"
                shutil.move(os.path.join(subdir, file), os.path.join(formatted_ann_folder, newFileName))

    print("DONE.")

    # create the annotations
    convert(BaseFolder)
