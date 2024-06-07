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

import matplotlib.pyplot as plt
import pickle
import sys
import os
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
import numpy as np


def bootstrap_roc(true_labels, predicted_scores, n_bootstrap=1):
    bootstrapped_tpr = []
    bootstrapped_fpr = []
    # Create dynamic thresholds based on quantiles and linear spacing
    quantile_values = np.quantile(predicted_scores, np.linspace(0, 1, 100))
    linear_values = np.linspace(predicted_scores.min(), predicted_scores.max(), 400)
    thresholds = np.unique(np.concatenate([quantile_values, linear_values]))  # Remove duplicates


    for i in range(n_bootstrap):
        print("Bootstrap N ", i + 1)
        # Bootstrap sampling
        indices = resample(np.arange(len(true_labels)))
        bs_labels = true_labels[indices]
        bs_scores = predicted_scores[indices]

        # Pre-calculate positives and negatives
        positives = bs_labels == 1
        negatives = bs_labels == 0
        pos_count = positives.sum()
        neg_count = negatives.sum()

        tprs = []
        fprs = []
        for threshold in thresholds:
            # Calculate TPR and FPR at each threshold using pre-calculated sums
            tpr = np.sum((bs_scores >= threshold) & positives) / pos_count
            fpr = np.sum((bs_scores >= threshold) & negatives) / neg_count
            tprs.append(tpr)
            fprs.append(fpr)

        bootstrapped_tpr.append(tprs)
        bootstrapped_fpr.append(fprs)

    # Calculate mean and 95% CI for TPR and FPR at each threshold
    mean_tpr = np.mean(bootstrapped_tpr, axis=0)
    lower_tpr = np.percentile(bootstrapped_tpr, 2.5, axis=0)
    upper_tpr = np.percentile(bootstrapped_tpr, 97.5, axis=0)

    mean_fpr = np.mean(bootstrapped_fpr, axis=0)

    return mean_fpr, mean_tpr, lower_tpr, upper_tpr, thresholds


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("Usage: python combined_ROC.py <path_to_real_colon_base> <path_to_test_dir> <output_folder>")
        sys.exit(1)

    output_folder = sys.argv[1]
    # list_of_pkls = ["list_of_pkl_files_output_from_standard_ROC/labels_scores.pkl"]

    # Initialize the plot
    plt.figure(figsize=(10, 7))

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']  # Color for each model
    model_labels = ['UCD', 'UCD+001-004', 'UCD + 004', 'UCD + 003', 'UCD + 002', 'UCD + 001']  # Labels for each model

    for pkl_path, color, label in zip(list_of_pkls, colors, model_labels):
        print("Working on: ", label)

        # Load the true_labels and predicted_scores from the pickle file
        with open(pkl_path, 'rb') as f:
            true_labels, predicted_scores = pickle.load(f)

        true_labels = true_labels[::50]
        predicted_scores = predicted_scores[::50]

        # Compute the bootstrapped ROC
        mean_fpr, mean_tpr, lower_tpr, upper_tpr, thresholds = bootstrap_roc(true_labels, predicted_scores)
        # Compute and print the AUC
        auc_score = roc_auc_score(true_labels, predicted_scores)

        # Save the ROC data to a pickle file
        roc_data = {
            'mean_fpr': mean_fpr,
            'mean_tpr': mean_tpr,
            'lower_tpr': lower_tpr,
            'upper_tpr': upper_tpr,
            'thresholds': thresholds,
            'auc_score': auc_score
        }
        save_path = os.path.join(output_folder, f'{label}_roc_data.pkl')
        with open(save_path, 'wb') as handle:
            pickle.dump(roc_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Construct the path to the pickle file
        load_path = os.path.join(output_folder, f'{label}_roc_data.pkl')

        # Open the pickle file and load the data
        with open(load_path, 'rb') as handle:
            roc_data = pickle.load(handle)

        # Now you can access the data from the loaded roc_data dictionary
        mean_fpr = roc_data['mean_fpr']
        mean_tpr = roc_data['mean_tpr']
        lower_tpr = roc_data['lower_tpr']
        upper_tpr = roc_data['upper_tpr']
        thresholds = roc_data['thresholds']
        auc_score = roc_data['auc_score']

        # Plot the ROC curve for this model
        plt.plot(mean_fpr[1:], mean_tpr[1:], label=f'{label}, AUC: {auc_score:.3f}', color=color)
        # plt.fill_between(mean_fpr, lower_tpr, upper_tpr, color=color, alpha=0.5)

    # Customize the plot
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.title(r'ROC on $RC_{MT}$', fontsize=22)
    plt.legend(title="Models", loc='lower right', fontsize=16, title_fontsize=18)

    # Set limits for x and y axes
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(True)

    # Save the figure
    plt.savefig(os.path.join(output_folder, 'comparative_roc_curve_with_bootstrapping.png'), dpi=300)

    print("DONE.")
