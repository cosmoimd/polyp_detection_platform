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
import sys


def is_frame_positive(file_path):
    """Determine if a frame is positive based on file content."""
    with open(file_path, 'r') as file:
        content = file.read().strip()
        return bool(content)


def process_directory(directory_path):
    """Process each .txt file in the directory, counting positive and negative frames."""
    print("Processing labels...")
    total_frames = 0
    positive_frames = 0
    negative_frames = 0

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            total_frames += 1
            file_path = os.path.join(directory_path, filename)
            if is_frame_positive(file_path):
                positive_frames += 1
            else:
                negative_frames += 1

    print("Done.")

    print(f'Positive Frames: {positive_frames}')
    print(f'Negative Frames: {negative_frames}')
    print(f'Total Frames: {total_frames}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python dataset_stats.py <path_to_labels_folder>")
        sys.exit(1)

    labels_folder = sys.argv[1]
    process_directory(labels_folder)
