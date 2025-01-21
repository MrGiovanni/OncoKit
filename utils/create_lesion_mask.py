"""
python create_lesion_mask.py --dataset_path /path/to/AbdomenAtlasPre \
                             --classification_labels_csv /path/to/abdomenatlas_classification_labels.csv \
                             --process_ids_txt /path/to/AbdomenAtlas2.0.txt >> lesion_mask_creation.log

1. Create a 0-mask lesion.nii.gz if doctors mark the case as "N"
2. Merge all ***_lesion.nii.gz, ***_tumor.nii.gz, ***_cyst.nii.gz (ignore pancreatic_pdac.nii.gz, pancreatic_cyst.nii.gz and pancreatic_pnet.nii.gz) into a single lesion.nii.gz
3. Do not create lesion.nii.gz if doctors have not annotated the case.

TODO: add functionality for multiprocessing
"""

import os
import argparse
import pandas as pd
import nibabel as nib
import numpy as np

def load_case_metadata(csv_file):
    """
    Load the csv file.
    
    Args:
        csv_file (str): path to the csv file.
    """
    return pd.read_csv(csv_file)

def load_ids_to_process(txt_file):
    """
    Load the IDs to process from a text file.
    
    Args:
        txt_file (str): path to the text file with IDs to process.
    """
    with open(txt_file, 'r') as file:
        ids = [line.strip() for line in file.readlines()]
    return ids

def create_empty_lesion_masks(segmentations_path, liver_img, row):
    """
    Create empty lesion masks for organs marked as 'N'.
    
    Args:
        segmentations_path (str): path to the "segmentations" folder.
        liver_img (nibabel.nifti1.Nifti1Image): liver.nii.gz for reference shape, affine, and header.
        row (pandas.Series): row from the classification data.
    """
    empty_data = np.zeros(liver_img.shape, dtype=np.uint8)
    for organ, mark in row.items():
        if organ != 'BDMAP_ID' and mark == 'N':
            organ_lesion_path = os.path.join(segmentations_path, f'{organ.lower()}_lesion.nii.gz')
            empty_img = nib.Nifti1Image(empty_data, liver_img.affine, liver_img.header)
            empty_img.set_data_dtype(np.uint8)
            nib.save(empty_img, organ_lesion_path)
            print(f"created empty mask: {organ_lesion_path}")

def get_lesion_files(segmentations_path):
    """
    Retrieve a list of lesion-related files to merge.
    
    Args:
        segmentations_path (str): path to the "segmentations" folder.
    """
    return [
        f for f in os.listdir(segmentations_path)
        if ('_lesion.nii.gz' in f or '_tumor.nii.gz' in f or '_cyst.nii.gz') and
           not (f.startswith('pancreatic_pdac') or f.startswith('pancreatic_cyst') or f.startswith('pancreatic_pnet')) and
           not f.startswith('_')
    ]

def merge_lesion_files(segmentations_path, liver_img, lesion_files):
    """
    Merge lesion-related files into a single lesion.nii.gz.
    
    Args:
        segmentations_path (str): path to the "segmentations" folder.
        liver_img (nibabel.nifti1.Nifti1Image): liver.nii.gz for reference shape, affine, and header.
        lesion_files (list): list of lesion-related files to merge.
    """
    combined_data = np.zeros(liver_img.shape, dtype=np.uint8)
    for lesion_file in lesion_files:
        lesion_path = os.path.join(segmentations_path, lesion_file)
        lesion_img = nib.load(lesion_path)
        if lesion_img.shape == combined_data.shape:
            combined_data = np.logical_or(combined_data, lesion_img.get_fdata()).astype(np.uint8)
        else:
            print(f"shape mismatch for {lesion_file}, skipping")
    
    combined_lesion_path = os.path.join(segmentations_path, 'lesion.nii.gz')
    combined_img = nib.Nifti1Image(combined_data, liver_img.affine, liver_img.header)
    combined_img.set_data_dtype(np.uint8)
    nib.save(combined_img, combined_lesion_path)
    print(f"created combined lesion file: {combined_lesion_path}")

def process_case(input_folder, case_id, classification_data):
    """
    Process a single case: create empty lesion masks and merge lesion files.
    """
    case_path = os.path.join(input_folder, case_id)
    segmentations_path = os.path.join(case_path, 'segmentations')
    liver_path = os.path.join(segmentations_path, 'liver.nii.gz')
    
    if not os.path.isdir(case_path):
        print(f"case directory not found: {case_id}, skipping")
        return
    
    if not os.path.isfile(liver_path):
        print(f"liver file not found for case {case_id}, skipping")
        return
    
    liver_img = nib.load(liver_path)
    create_empty_lesion_masks(segmentations_path, liver_img, classification_data.loc[classification_data['BDMAP_ID'] == case_id].iloc[0])

    lesion_files = get_lesion_files(segmentations_path)
    if lesion_files:
        merge_lesion_files(segmentations_path, liver_img, lesion_files)
    else:
        print(f"no lesion files found for case {case_id}, skipping lesion.nii.gz creation.")

def process_abdomenatlas(input_folder, csv_file, ids_txt):
    """
    Process selected cases in the AbdomenAtlas dataset.
    """
    classification_data = load_case_metadata(csv_file)
    selected_ids = load_ids_to_process(ids_txt)

    for case_id in selected_ids:
        if case_id in classification_data['BDMAP_ID'].values:
            process_case(input_folder, case_id, classification_data)
        else:
            print(f"case ID {case_id} not found in classification data, skipping")

def main():
    parser = argparse.ArgumentParser(description="process AbdomenAtlas cases.")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the AbdomenAtlasPre folder.")
    parser.add_argument("--classification_labels_csv", type=str, required=True, help="path to the classification labels csv file.")
    parser.add_argument("--process_ids_txt", type=str, required=True, help="path to the txt file with IDs to process.")
    args = parser.parse_args()
    
    process_abdomenatlas(args.dataset_path, args.classification_labels_csv, args.process_ids_txt)

if __name__ == "__main__":
    main()
