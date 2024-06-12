# Open-Access Polyp Detection Platform

This repository supports the paper "Assessing Clinical Efficacy of Polyp Detection Models Using Open-Access Datasets." It includes scripts for downloading, formatting, and preparing five open-access datasets, along with tools for generating ROC curves, event-based metrics, and FROC plots.

## Repository Structure

- **dataset_downloads**: Scripts to download, format, and prepare datasets.
- **metrics**: Scripts for generating per-box and per-frame ROC, event-based metrics, and FROC plots.

Each directory includes a README with specific usage instructions.

## Installation

Clone the repository and follow these steps to set up the necessary environment:

1. **Install Python**: Ensure that Python 3.8 or later is installed on your system.

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. **Install dependencies using requirements file** 
   ```bash
   pip install -r requirements.txt`

## Citation

If you find this repository useful in your research, please consider citing our work:

> Gabriel Marchese Aizenman, Pietro Salvagnini, Andrea Cherubini, and Carlo Biffi, "Assessing Clinical Efficacy of Polyp Detection Models Using Open-Access Datasets".

## License
```plaintext
# Copyright (C) 2024 Cosmo Intelligent Medical Devices
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
