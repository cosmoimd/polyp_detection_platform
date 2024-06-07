#!/bin/bash

# Check if one argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <ldpolyp_directory>"
    exit 1
fi

# Assign the argument to a variable
BASE_DIR=$1

# Set the AVI directory and output directory based on the base directory
AVI_DIR="${BASE_DIR}/videos without polyps"
OUTPUT_DIR="${BASE_DIR}/extracted_negative_frames"

# Check if the AVI directory exists
if [ ! -d "$AVI_DIR" ]; then
    echo "Directory not found: $AVI_DIR"
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through each AVI file in the directory
for VIDEO_FILE in "$AVI_DIR"/*.avi; do
    # Skip if not a file
    if [ ! -f "$VIDEO_FILE" ]; then
        continue
    fi

    # Extract file name without extension
    FILENAME=$(basename "$VIDEO_FILE" .avi)

    # Extract frames - one frame every 5 frames
    ffmpeg -i "$VIDEO_FILE" -vf "select=not(mod(n\,5))" -vsync vfr -q:v 2 "$OUTPUT_DIR/${FILENAME}_frame_%04d.jpg"
done

echo "Frame extraction complete."