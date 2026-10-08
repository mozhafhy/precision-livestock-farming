# Precision Livestock Farm

## Overview

This is the main repository for chicken fecal segmentation. The purpose of the project is to compare segmentation models: YOLOv8-seg and U-Net (+MobileNetV2). This project uses two stages model: one to make ROI detector model and one to create the segmentation model.

The datasets used to train, validate, and test the model are sourced from:
1. Kaggle: https://www.kaggle.com/datasets/allandclive/chicken-disease-1
2. Mendeley: ...
3. Internally collected fecal images

## Project Structure

```sh
.
├── dataset/
│   ├── annotations/
│   │   ├── _annotations.coco.json # original COCO JSON from Roboflow
│   │   ├── master_cleaned.json
│   │   ├── README.dataset.txt
│   │   └── README.roboflow.txt
│   ├── metadata/
│   │   └── sources.csv
│   ├── raw/ # images used to train the models
│   │   ├── KGL_0000.jpg
│   │   ├── KGL_0001.jpg
│   │   ├── ...
│   │   └── <SOURCE>_<ID>.jpg
│   ├── split_meta.json
│   └── split.csv
├── scripts/
│   ├── 01_clean_and_generate_split_csv.py
│   ├── 02_split_annotations.py
│   └── generate_sources_csv.sh
├── utils/
│   ├── __init__.py
│   ├── file_managements.py
│   ├── preprocessing.py
│   ├── types.py
│   └── visualization.py
├── .gitignore
├── chores.ipynb
├── environment.yml
├── LICENSE
├── main.ipynb
├── README.md
└── work_separator.py
```

The `master_cleaned.json` is generated from `_annotations.coco.json` to match the cleaned and renamed image filenames in `raw/`. The original filenames from Roboflow are like: `<arbitrary-name>.rf.<hash-code>.jpg`. We tranformed those name to this format: `<SOURCE>_<ID>.jpg`. The `<SOURCE>` is defined manually in `dataset/metadata/sources.csv` for each image.

## How to Run the Project

### Notes

This project uses Conda as package manager. Make sure you have Anaconda or Miniconda installed and configured in your computer.

### Creating the Environment

Open your terminal and run this command to create Conda environment prefixed in the current working directory:

```sh
conda env create -f environment.yml --prefix ./env
```

### Activating the Environment

Run this command in your terminal to activate the Conda environment:

```sh
conda activate ./env
```

### Running the `scripts/`

The `scripts/` directory contains two Python scripts and one bash script:

1. `01_clean_and_generate_split_csv.py`
    ```
    python -m scripts.01_clean_and_generate_split_csv
    ```

    This script is used to clean and tranform the image filename from Roboflow's default filename to `<SOURCE>_<ID>.jpg`. Then, generates the `split.csv`, `master_cleaned.json`, and `split_meta.json`. Requires `sources.csv` to run properly. It is recommended to run `generate_sources_csv.sh` before and after executing this script.

2. `02_split_annotations.py`
    ```
    python -m scripts.02_split_annotations
    ```

    This script is used to split the master JSON into train, test, and validation JSON based on the `split.csv`.

3. `generate_sources_csv.sh`
    ```
    bash scripts/generate_sources_csv.sh
    ```

    This script is used to generate the `source.csv`

The scripts execution order should be:

`generate_sources_csv.sh` -> `01_clean_and_generate_split_csv.py` -> `02_split_annotations.py`

## The `dataset/`

If you are willing to use different images and annotations, you can put your Roboflow images in the `dataset/raw/` and the annotation JSON in `dataset/annotations/`. Then, you can run the scripts as mentioned in the "Running the `scripts/`" section to generate your own `sources.csv`, `split_meta.json`, cleaned annotation COCO JSON.

## Additional Information

The project is still incomplete. This is just a minimal README to get the hunch of the project. More complete information will be added soon following the progress of this project.
