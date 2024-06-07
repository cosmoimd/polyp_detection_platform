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
import pickle as pkl
import sys
import os


def plot_multiple_models_froc(all_conf_results, save_file, legend_names, legend_title):
    """
    Create a plot of the standard fROC for multiple studies, with a legend indicating the study name.

    Args:
    - all_conf_results (dict): Dictionary with keys as study names and values as dictionaries of data points for the
    standard fROC.
    - save_file (str): Path including filename where the plot will be saved.
    - legend_names (list): List of custom legend names corresponding to the study names. If not provided, the study
    names will be split and the first part used.
    - legend_title (str): Custom title for the legend. If empty, defaults to "Legend".

    Returns:
    - None. This function generates a file with an fROC plot.
    """
    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a list of colors for visual distinction between studies
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']
    if len(all_conf_results) > len(colors):
        print("Warning: Not enough predefined colors for all studies. Colors will repeat.")

    # Loop through each study for the standard fROC plot
    for study_index, (study_name, data) in enumerate(all_conf_results.items()):
        fpe_values = [values['fpr'] for _, values in data.items()]
        rt_values = [values['frames_rt'] for _, values in data.items()]
        color = colors[study_index % len(colors)]
        ax.plot(fpe_values, rt_values, color=color,
                label=legend_names[study_index] if legend_names else study_name.split("-")[0])

    for study_index, (study_name, data) in enumerate(all_conf_results.items()):
        # Check for the specific key 0.6 and mark with 'x'
        for key, values in data.items():
            if key == 0.6:
                ax.plot(values['fpr'], values['frames_rt'], 'x', color=colors[study_index % len(colors)],
                        markersize=7, markeredgewidth=1.5)

    ax.set_title('FROC Varying Detection Threshold \u03B4', fontsize=22)
    ax.set_xlabel('Avg FP Events Per-Video [#]', fontsize=20)
    ax.set_ylabel('Recall Per-Polyp [%]', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0)

    # Place the legend inside the plot
    ax.legend(title=legend_title if legend_title else "Legend", loc='lower right', fontsize=16, title_fontsize=18)

    # Adjust layout to make room for the legend without clipping
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved to {save_file}")


def plot_multiple_models_time_froc(all_time_results, save_file, legend_names, legend_title):
    """
    Create a plot of the standard fROC for multiple studies over varying time thresholds,
    with a legend indicating the study name.

    Args:
    - all_time_results (dict): Dictionary with keys as study names and values as dictionaries of data points for the
    standard fROC over varying time thresholds.
    - save_file (str): Path including filename where the plot will be saved.
    - legend_names (list): List of custom legend names corresponding to the study names. If not provided, the study
    names will be split and the first part used.
    - legend_title (str): Custom title for the legend. If empty, defaults to "Legend".

    Returns:
    - None. This function generates a file with an fROC plot over varying time thresholds.
    """
    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a list of colors for visual distinction between studies
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']
    if len(all_time_results) > len(colors):
        print("Warning: Not enough predefined colors for all studies. Colors will repeat.")

    # Loop through each study for the standard fROC plot
    for study_index, (study_name, data) in enumerate(all_time_results.items()):
        fpe_values = [values['fpr'] for _, values in data.items()]
        rt_values = [values['frames_rt'] for _, values in data.items()]
        color = colors[study_index % len(colors)]
        ax.plot(fpe_values, rt_values, color=color,
                label=legend_names[study_index] if legend_names else study_name.split("-")[0])


    # Check for the specific key 0.6 and mark with 'x'
    for study_index, (study_name, data) in enumerate(all_time_results.items()):
        for key, values in data.items():
            if key == 1:
                ax.plot(values['fpr'], values['frames_rt'], 'x', color=colors[study_index % len(colors)], markersize=7, markeredgewidth=1.5)

    ax.set_title('FROC Varying Event Length Threshold \u03C4', fontsize=22)
    ax.set_xlabel('Avg FP Events Per-Video [#]', fontsize=20)
    ax.set_ylabel('Recall Per-Polyp [%]', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0)

    # Place the legend inside the plot, adjusting its location as needed
    ax.legend(title=legend_title if legend_title else "Legend", loc='lower right', fontsize=16, title_fontsize=18)

    # Adjust layout to prevent clipping of tick-labels and accommodate the legend
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved to {save_file}")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python plot_fROC_pkls.py <save_path> <file_name> <pkl_file_1> <pkl_file_2> ...")
        sys.exit(1)

    save_path = sys.argv[1]
    file_name = sys.argv[2]
    pkl_paths = sys.argv[3:]

    # Change this if you want names for the plots other than the models
    legend_names = []
    legend_title = ""

    if len(pkl_paths) > 1:
        all_conf_results = {}
        all_time_results = {}

        # Load and add each graph pkl to the list
        for pkl_path in pkl_paths:
            base_path = os.path.basename(pkl_path)
            model_name = os.path.splitext(base_path)[0]
            with open(pkl_path, 'rb') as file:
                data = pkl.load(file)
                all_conf_results[model_name] = data[0]
                all_time_results[model_name] = data[1]

        save_file = os.path.join(save_path, f"{file_name}_det.png")
        plot_multiple_models_froc(all_conf_results, save_file, legend_names, legend_title)
        save_file = os.path.join(save_path, f"{file_name}_tau.png")
        plot_multiple_models_time_froc(all_time_results, save_file, legend_names, legend_title)
