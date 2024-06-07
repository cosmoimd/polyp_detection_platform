# Real Colon Scripts

## Export Yolo Format
This allows you to convert the real colon dataset into yolo format. You set the `base_dataset_folder`
and the `output_folder`, and optionally change the segmentation for training, validation, and testing
sets. You can also change the number of frames per lesion and negative ratio, then run the code with
`python export_yolo_format.py`.

## Convert Coco To Yolo
This creates a dataset in yolo format from an existing coco dataset. To use, change `base_coco_folder`
and `output_folder` then run `python convert_coco_to_yolo.py`

## Split UCD
This takes the unified colonoscopy dataset and splits it to use PolypGen as the validation set and the
rest as the training set. To use, change `base_ucd_folder`, `output_train_folder`, and `output_val_folder`
then run `python split_ucd.py`

## Study Dataset Creations
This creates datasets to test on each study in real colon, as well as on the whole dataset, and generates
coco json files for the evaluation. It also creates training splits for each study combined with UCD. 
To run, change `base_dataset_folder`, `ucd_path`, and `output_folder` then run
`python study_dataset_creations.py`

## Create Coco Annotations in JSON
This will create a json file for each of the datasets generates. It is automatically executed during the 
dataset creations, but if you want to run it again change the `base_dataset_folder` and `output_folder`
then run `python create_coco_ann_jsons.py`

## Filter Annotations
This code will create test datasets that include the first 1 second and 3 seconds of each polyp. To use, 
run `python filter_annotations.py <path_to_real_colon_base> <yolo_datasets_dir>`. Study_dataset_creations.py
must be completed before running this.

## Start to End Polyps
This code creates a dataset including all positive frames, and all negative frames in between positive frames
for a lesion. To use, change `base_dataset_folder` and `output_folder`, then run `python start_to_end_dataset.py`.
