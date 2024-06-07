# Colonoscopy Scripts
Enhance your polyp detection models with the Real Colon Scripts toolkit, designed for comprehensive evaluation
through reaction time analysis, accuracy metrics, missed detection identification, and performance visualization.

## ROC
This will calculate the plot of recall per-polyp (%) in 3 seconds of entering the frame
versus the fp events per video. It will also save a pkl file of the results that can be used to graph
multiple fROCs together using the `plot_fROC_pkls.py` code.
#### Usage:
`python standard_ROC.py <path_to_real_colon_base> <path_to_prediction_labels> <test_on> <output_path>`. 

## fROC
This will calculate the plot of recall per-polyp (%) in 3 seconds of entering the frame
versus the fp events per video. It will also save a pkl file of the results that can be used to graph
multiple fROCs together using the `plot_fROC_pkls.py` code.
#### Usage:
`python fROC.py <path_to_real_colon_base> <path_to_prediction_labels> <test_on>`. You can also modify variables 
such as `conf_check`, `iou_thr`, `tau_check`, and `sigma`.

## Plot fROC PKLs
This will create a plot combining different pkl files from the output of `fROC.py` code into one plot.
#### Usage:
`python plot_fROC_pkls.py <path_to_save_directory> <file_name> <pkl_file_1> <pkl_file_2> ...`. You can also change the
values of `legend_title` and `legend_names` if you want custom values.


### Parameter Legend
- `<path_to_real_colon_base>`: Path to the folder containing the real colon dataset downloads.
- `<path_to_prediction_labels>`: Path to the folder named 'labels' created after running a test. The folder should be 
saved in the directory 'runs/test/<test_name>'.
- `<test_on>`: The REAL-Colon split used for this test. Values can be 'study_001', 'study_002', 'study_003', 'study_004',
'all', or 'all_min_train'. Can also be 'start_to_end_polyp' or 'start_to_end_polyp_min_train' for `fROC.py`.
- `<path_to_save_directory>`: Path to the directory you would like to save the graph to.
- `<file_name>`: Name of the file you would like to save.
- `<pkl_file_x>`: Path to the .pkl file saved from running `fROC.py`.
