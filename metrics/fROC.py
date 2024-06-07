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

from parsing_helper import frame_parser
import matplotlib.pyplot as plt
import pickle as pkl
import sys
import os
import pandas as pd
import numpy as np


def calc_rt_and_fpr(all_polyp, all_detected, all_pred, basepath, sigma, time_check, conf_check, iou_thr, ste_polyp):
    """
    Calculates reaction times (RT) and false positive rates (FPR) for different detection methods
    based on varying confidence and IoU thresholds. Additionally, evaluates the impact of varying
    the time threshold on RT and FPR.

    Parameters:
    - all_polyp (dict): Maps polyps to their frame appearances.
    - all_detected (dict): Maps detected polyps to frames with confidence and IoU scores.
    - all_pred (dict): Maps frame names to predictions (confidence and IoU scores).
    - basepath (str): Path to directory containing 'video_info.csv' for fps data.
    - time_check (float): Time threshold for evaluating detection performance in seconds.
    - conf_check (float): Chosen confidence threshold for analysis.
    - iou_thr (float): Chosen IoU threshold for analysis.
    - ste_polyp (bool): Flag to consider start-to-end polyp detection for analysis.

    Returns:
    - dict: A dictionary containing calculated RTs and FPRs for varying
      confidence and IoU thresholds, segmented by the specified time threshold.
    """
    # Initialize final variables
    time_results = {round(i, 1): {} for i in np.arange(0.2, 3.0, 0.2)}

    # Creating fps dictionary to hold fps of each polyp/video
    video_info = pd.read_csv(os.path.join(basepath, 'video_info.csv'))
    fps_dict = {}
    video_fps_dict = {row['unique_video_name']: row['fps'] for index, row in video_info.iterrows()}

    for unique_id in all_polyp.keys():
        fps = video_info[video_info['unique_video_name'] == unique_id.split('_')[0]]['fps'].values[0]
        fps_dict[unique_id] = fps

    # Iterate through each time in the time_results dictionary and calculate results
    for tau in time_results.keys():
        print(f"Calculating results for time {tau}s...")
        # If time_check, then record the full confidence threshold range
        if tau == time_check:
            conf_results = {round(i, 1): {} for i in np.arange(0.1, 1.0, 0.1)}
        else:
            conf_results = {conf_check: {}}

        # Calculating FPR ########################################
        # Iterate through each confidence threshold in conf_results dict
        for conf_thr in conf_results.keys():
            print(f"Calculating results for confidence {conf_thr}...")

            # Define variables
            consec_fp = 0
            fpr_count = 0
            norm_fpr_count = 0
            curr_fps = 0
            curr_vid = ""
            tot_vids = 0
            tot_neg_frames = 0
            all_pol_list = []

            # Iterate through each frame of predictions
            for frame_name in all_pred.keys():

                # If a new video, update variables
                if frame_name.split("_")[0] != curr_vid:
                    tot_vids += 1
                    curr_vid = frame_name.split("_")[0]
                    curr_fps = tau * video_fps_dict[curr_vid]
                    # If all polyps with negatives, make the list to hold all the frame numbers
                    if ste_polyp:
                        all_pol_list = []
                        for i in all_polyp.keys():
                            if i.split("_")[0] == curr_vid:
                                for x in range(all_polyp[i][0], all_polyp[i][len(all_polyp[i]) - 1] + 1):
                                    all_pol_list.append(x)

                if ste_polyp:
                    if int(float(frame_name.split('_')[1])) not in all_pol_list:
                        continue

                if all_pred[frame_name]:
                    predictions = all_pred[frame_name]
                    conf_reached = False
                    is_neg = False

                    for confidence, iou in predictions:
                        if iou == -1:
                            is_neg = True
                        if confidence > conf_thr:
                            conf_reached = True

                    if is_neg:
                        tot_neg_frames += 1
                        if not conf_reached:
                            # True negative
                            consec_fp = 0
                        else:
                            # False positive
                            consec_fp += 1
                else:
                    # true negative
                    consec_fp = 0
                    tot_neg_frames += 1

                if consec_fp >= curr_fps:
                    fpr_count += 1
                    norm_fpr_count += consec_fp
                    consec_fp = 0

            # Calculate and write overall metrics
            conf_results[conf_thr]['fpr'] = round(fpr_count / tot_vids, 3)
            conf_results[conf_thr]['norm_fpr'] = round(fpr_count / tot_neg_frames, 5)

            all_det = {}
            for unique_id, frames in all_detected.items():
                # Check and save frames for which both confidence and IoU meet their respective thresholds
                all_det[unique_id] = [frame_number for frame_number, (confidence, iou) in
                                      sorted(frames.items()) if
                                      confidence > conf_thr and iou > iou_thr]

            # Find reaction time given the dictionaries
            detected_polyps = 0
            frame_cutoff = 0

            for polyp_id in all_polyp.keys():
                frame_count = 0

                if polyp_id.split("_")[0] != curr_vid:
                    curr_vid = polyp_id.split("_")[0]
                    curr_fps = tau * video_fps_dict[curr_vid]
                    frame_cutoff = sigma * video_fps_dict[curr_vid]

                consec_det = 0
                if polyp_id in all_det.keys():
                    dets = all_det[polyp_id]
                else:
                    dets = []

                for frame in all_polyp[polyp_id]:
                    if frame in dets:
                        consec_det += 1
                    else:
                        consec_det = 0

                    if consec_det >= curr_fps:
                        detected_polyps += 1
                        break

                    frame_count += 1
                    if frame_count == frame_cutoff:
                        break

            conf_results[conf_thr]['frames_rt'] = round(detected_polyps / len(list(all_polyp.keys())), 3)

        time_results[tau] = conf_results
    return time_results


def plot_conf_froc(data, time_thr, output_dir):
    """
    Plot and save a combined graph of reaction time vs false positive rate and
    reaction time vs normalized false positive rate for a given time threshold.

    Args:
    - data (dict): The dictionary containing the data for plotting.
    - time_thr (int or float): The specific time threshold to plot.
    - output_dir (str): The directory path where the plot image will be saved.

    Returns:
    - None
    """
    # Data for the standard fROC plot
    specific_time_data = data.get(time_thr, {})
    fpr_values, rt_values = [], []
    for _, values in specific_time_data.items():
        fpr_values.append(values['fpr'])
        rt_values.append(values['frames_rt'])

    # Data for the normalized fROC plot
    norm_fpr_values, norm_rt_values = [], []
    for _, values in specific_time_data.items():
        norm_fpr_values.append(values.get('norm_fpr', values['fpr']))  # Fallback to 'fpr' if 'norm_fpr' is missing
        norm_rt_values.append(values['frames_rt'])

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # 20x6 figure with two plots

    # Plot standard fROC
    axs[0].plot(fpr_values, rt_values, color='blue')
    for confidence, values in specific_time_data.items():
        axs[0].text(values['fpr'], values['frames_rt'], f'{confidence}', fontsize=9)
    axs[0].set_title('fROC Varying Detection Threshold')
    axs[0].set_xlabel('Avg FP Events Per-Video [#]')
    axs[0].set_ylabel('Recall Per-Polyp [%]')
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(left=0)
    axs[0].grid(True)

    # Plot normalized fROC
    axs[1].plot(norm_fpr_values, norm_rt_values, color='blue')
    for confidence, values in specific_time_data.items():
        axs[1].text(values.get('norm_fpr', values['fpr']), values['frames_rt'], f'{confidence}', fontsize=9)
    axs[1].set_title('Normalized fROC Varying Detection Threshold')
    axs[1].set_xlabel('Avg FP Events Per-Video [#]')
    axs[1].set_ylabel('Recall Per-Polyp [%]')
    axs[1].set_xlim(0, 0.001)
    axs[1].set_ylim(0, 1)
    axs[1].grid(True)

    # Save the figure to a file
    filename = os.path.join(output_dir, f"fROC_at_{time_thr}s.png")
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Graph saved to {filename}")


def plot_time_froc(data, conf_thr, output_dir):
    """
    Plot and save a combined graph of reaction time vs false positive rate and
    reaction time vs normalized false positive rate for a given time threshold.

    Args:
    - data (dict): The dictionary containing the data for plotting.
    - time_thr (int or float): The specific time threshold to plot.
    - output_dir (str): The directory path where the plot image will be saved.

    Returns:
    - None
    """
    # Data for the standard fROC plot
    fpr_values, rt_values = [], []
    for _, values in data.items():
        fpr_values.append(values['fpr'])
        rt_values.append(values['frames_rt'])

    # Data for the normalized fROC plot
    norm_fpr_values, norm_rt_values = [], []
    for _, values in data.items():
        norm_fpr_values.append(values.get('norm_fpr', values['fpr']))  # Fallback to 'fpr' if 'norm_fpr' is missing
        norm_rt_values.append(values['frames_rt'])

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # 20x6 figure with two plots

    # Plot standard fROC
    axs[0].plot(fpr_values, rt_values, color='blue')
    for time, values in data.items():
        axs[0].text(values['fpr'], values['frames_rt'], f'{time}', fontsize=9)
    axs[0].set_title('FROC Varying \u03C4')
    axs[0].set_xlabel('Avg FP Events Per-Video [#]')
    axs[0].set_ylabel('Recall Per-Polyp [%]')
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(left=0)
    axs[0].grid(True)

    # Plot normalized fROC
    axs[1].plot(norm_fpr_values, norm_rt_values, color='blue')
    for time, values in data.items():
        axs[1].text(values.get('norm_fpr', values['fpr']), values['frames_rt'], f'{time}', fontsize=9)
    axs[1].set_title('Normalized FROC Varying \u03C4')
    axs[1].set_xlabel('Avg FP Events Per-Video [#]')
    axs[1].set_ylabel('Recall Per-Polyp [%]')
    axs[1].set_xlim(0, 0.007)
    axs[1].set_ylim(0, 1)
    axs[1].grid(True)

    # Save the figure to a file
    filename = os.path.join(output_dir, f"fROC_at_det_{conf_thr}.png")
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Graph saved to {filename}")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python fROC.py <path_to_real_colon_base> <path_to_prediction_dir> <test_on>")
        sys.exit(1)

    basepath = sys.argv[1]
    pred_path = sys.argv[2]
    test_on = sys.argv[3]  # Should be study_00<study_num>, all, or test_set or start_to_end_polyp or with _min_train

    # Check for start to end polyp options
    ste_polyp = False
    if test_on == "start_to_end_polyp":
        test_on = "all"
        print("testing on start_to_end_polyp")
        ste_polyp = True
    elif test_on == "start_to_end_polyp_min_train":
        test_on = "test_set"
        ste_polyp = True
        print("testing on start_to_end_polyp_min_train")

    # Set parameters
    tau_check = 2
    conf_check = 0.6
    iou_thr = 0.2
    sigma = 3
    output_dir = os.path.abspath(os.path.join(pred_path, os.pardir))

    # Get statistics for each frame and object for reaction times
    all_polyp, all_detected, all_pred = frame_parser(basepath, pred_path, test_on)

    # Calculate the results dictionary to get all confidence threshold results
    results = calc_rt_and_fpr(all_polyp, all_detected, all_pred, basepath, sigma,
                              tau_check, conf_check, iou_thr, ste_polyp)

    # Create a dictionary to hold the time change results
    tot_time_dict = {}
    for time, res in results.items():
        tot_time_dict[time] = res[conf_check]

    # Create graphs for the results
    plot_conf_froc(results, tau_check, output_dir)
    plot_time_froc(tot_time_dict, conf_check, output_dir)

    # Save pkl file for plotting with other models later
    model_name = os.path.basename(output_dir)
    pkl_file = os.path.join(output_dir, f"{model_name}_in_{sigma}s_froc.pkl")
    with open(pkl_file, 'wb') as file:
        pkl.dump([results[tau_check], tot_time_dict], file)
    print(f"Saved pkl file to {pkl_file}")
    print("DONE.")
