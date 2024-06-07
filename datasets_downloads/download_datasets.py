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
import zipfile
import rarfile
import subprocess
import synapseclient


def ensure_dir(directory):
    """
    Creates a directory if it does not exist.

    Args:
        directory (str): Path to the directory to ensure its existence.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def extract_file(path, to_directory):
    """
    Extracts a file to a specified directory, supporting both .zip and .rar files.

    Args:
        path (str): Path to the .zip or .rar file to be extracted.
        to_directory (str): Target directory where the contents will be extracted.
    """
    ensure_dir(to_directory)
    if path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            print(f"Unzipping {path}...")
            zip_ref.extractall(to_directory)
    elif path.endswith('.rar'):
        with rarfile.RarFile(path) as rar_ref:
            print(f"Unzipping {path}...")
            rar_ref.extractall(to_directory)


def download_from_dataverse(download_path):
    """
    Downloads and extracts a dataset from Dataverse to a specified path.

    Args:
        download_path (str): Directory where the downloaded dataset will be extracted.
    """
    print("Downloading KUMC dataset...")
    output_zip = os.path.join(download_path, "dataset.zip")
    curl_command = f"curl -L https://dataverse.harvard.edu/api/access/dataset/:persistentId?persistentId=doi:10.7910/DVN/FCBUOR --output {output_zip}"
    try:
        subprocess.run(curl_command, check=True, shell=True)
        print(f"File downloaded successfully: {output_zip}")
        extract_file(output_zip, download_path)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while downloading the file: {e}")


def download_from_synapse(auth_token, entity_id, download_folder):
    """
    Downloads and extracts a dataset from Synapse to a specified folder.

    Args:
        auth_token (str): Authentication token for Synapse.
        entity_id (str): Entity ID of the dataset to download.
        download_folder (str): Directory where the dataset will be downloaded and extracted.
    """
    print("Downloading PolypGen dataset...")
    ensure_dir(download_folder)
    syn = synapseclient.Synapse()
    syn.login(authToken=auth_token)
    entity = syn.get(entity=entity_id, downloadLocation=download_folder)
    print(f"Downloaded PolypGen dataset to {download_folder}")
    extract_file(download_folder, download_folder)


def main(base_path, syn_auth_token, ldp_path):
    """
    Manages the download, extraction, and combination of multiple datasets into a unified COCO format.

    Args:
        base_path (str): Base directory for storing the combined COCO dataset and related files.
        syn_auth_token (str): Authentication token for downloading datasets from Synapse.
        ldp_path (str): Path containing local .zip or .rar files of the LDPolyp dataset for extraction.

    Returns:
        None. Creates a COCO-formatted JSON file and organizes related images in the specified directory.
    """

    ensure_dir(base_path)

    # Synapse Dataset
    synapse_folder = os.path.join(base_path, "PolyGenDataset")
    download_from_synapse(auth_token=syn_auth_token, entity_id="syn45200214", download_folder=synapse_folder)

    # Dataverse Dataset
    dataverse_folder = os.path.join(base_path, "KUMCDataset")
    ensure_dir(dataverse_folder)
    download_from_dataverse(download_path=dataverse_folder)

    # LDPolyp Dataset unzip
    for filename in os.listdir(ldp_path):
        file_path = os.path.join(ldp_path, filename)
        if file_path.endswith('.zip') or file_path.endswith('.rar'):
            print(f"Found a compressed file: {file_path}")
            extract_file(file_path, ldp_path)

    print("DONE.")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python download_datasets.py <path_to_download> <path_to_LDPolyp_dataset_zips>")
        sys.exit(1)

    base_path = sys.argv[1]
    ldp_path = sys.argv[2]

    if not os.path.exists(ldp_path):
        print(f"The folder {ldp_path} does not exist.")
        sys.exit(1)

    syn_auth_token = ""
    main(base_path, syn_auth_token, ldp_path)
