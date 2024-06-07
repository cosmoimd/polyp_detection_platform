# Dataset Formatting for COCO

This repository provides a script to format different open access colonoscopy datasets to COCO format.

## Datasets Downloads
First, download the zip files of LDPolyp Dataset from https://github.com/dashishi/LDPolypVideo-Benchmark into a directory.
Once that is done, create a synapse authentication token and change the value of the syn_auth_token
in colonoscopy_data/download_datasets.py. Then run `python colonoscopy_data/download_datasets.py <path_to_download> 
<path_to_LDPolyp_dataset_zips>`. This will download the other two datasets, and unzip all the 
compressed files. (SUN and REAL Colon datasets downloaded separately)

## Datasets Used
Here are the datasets used, along with the download links and instructions to
use the scripts to convert the datasets to coco format. 

#### PolypGen Dataset from 2023
* Download link: https://www.synapse.org/#!Synapse:syn26376615/wiki/613312
* To convert to coco format, change the BaseFolder in colonoscopy_data/convert_polypgen_to_coco.py,
and then run `python colonoscopy_data/convert_polypgen_to_coco.py`. This will create
a coco folder to contain all the images in train_images and train_ann.json for annotations.

#### LDPolypVideo Dataset from 2021
* Download link: https://github.com/dashishi/LDPolypVideo-Benchmark
* Extract negative images from the .avi videos with no polyps: 
`./extract_negative_frames.sh /path/to/ldpolyp/dir`. This will iterate 
through the videos and save one image every 5 frames to a extracted_negative_frames folder.
* To convert to coco format, change the BaseFolder path in colonoscopy_data/ldpolyp_reformat.py,
and then run `python colonoscopy_data/ldpolyp_reformat.py`. This will create
a coco folder to contain all the images in train_images and train_ann.json for annotations.

#### KUMC Dataset from 2021
* Download link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FCBUOR
* To convert to coco format, change the BaseFolder in colonoscopy_data/kumc_reformat.py,
and then run `python colonoscopy_data/kumc_reformat.py`. This will create
a coco folder to contain all the images in train_images and train_ann.json for annotations.

#### SUN Dataset from 2021
* Download link: http://amed8k.sundatabase.org/
* To convert to coco format, change the BaseFolder in colonoscopy_data/sun_reformat.py,
and then run `python colonoscopy_data/sun_reformat.py`. This will create
a coco folder to contain all the images in train_images and train_ann.json for annotations.

#### REAL Colon Dataset from 2024
* Download link:  https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866
* Run `python figshare_dataset.py` to automatically download the dataset in full from Figshare to the 
./dataset folder. Output folder can be updated setting variable `DOWNLOAD_DIR` in `figshare_dataset.py`.

## Dataset Combination 
In order to create a combined dataset using symbolics please make sure to download and convert
each dataset to coco format using the instructions above. Then, adjust any paths necessary in combine_coco_datasets.py
and run `python colonoscopy_data/combine_coco_datasets.py`. A new folder will be made with train_ann.json and 
train_images containing all the images from each dataset combined.
