"""
python create_lesion_masks.py --dataset_path /data2/wenxuan/Project/AbdomenAtlasDatasetProfiler/AbdomenAtlasPre --classification_labels_csv ./abdomenatlas_classification_labels.csv --process_ids_txt ./case_id/AbdomenAtlas2.0.txt --max_workers 20 --skip_existing >> ./logs/lesion_masks_creation.log

Explanation for all arguments:
--dataset_path: change path to where the AbdomenAtlas folder stored
--classification_labels_csv: no need to change, already the relative path
--process_ids_txt: no need to change, already the relative path
--max_workers: number of workers to use for multiprocessing
--skip_existing: skip cases with existing lesion.nii.gz files

!!!Caution: Removing "--skip_existing" will overwrite existing lesion.nii.gz files for all cases in the process_ids_txt file.

1. Create a 0-mask lesion.nii.gz if doctors mark the case as "N"
2. Merge all ***_lesion.nii.gz, ***_tumor.nii.gz, ***_cyst.nii.gz (ignore pancreatic_pdac.nii.gz, pancreatic_cyst.nii.gz and pancreatic_pnet.nii.gz) into a single lesion.nii.gz
3. Do not create lesion.nii.gz if doctors have not annotated the case.

"""

import os
import argparse
import pandas as pd
import nibabel as nib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

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
            empty_img.get_data_dtype(finalize=True)
            nib.save(empty_img, organ_lesion_path)
            print(f"created empty mask: {organ_lesion_path}", flush=True)

def get_lesion_files(segmentations_path):
    """
    Retrieve a list of lesion-related files to merge.
    Only includes files ending with _lesion.nii.gz, _tumor.nii.gz, or _cyst.nii.gz.
    Excludes files starting with "_" and specific pancreatic lesion files.
    
    Args:
        segmentations_path (str): path to the "segmentations" folder.
    """
    return [
        f for f in os.listdir(segmentations_path)
        if (
            f.endswith('_lesion.nii.gz') or f.endswith('_tumor.nii.gz') or f.endswith('_cyst.nii.gz')
        ) and not (
            f.startswith('_') or
            f in {'pancreatic_pdac.nii.gz', 'pancreatic_cyst.nii.gz', 'pancreatic_pnet.nii.gz'}
        )
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
            print(f"shape mismatch for {lesion_file}, skipping", flush=True)
    
    combined_lesion_path = os.path.join(segmentations_path, 'lesion.nii.gz')
    combined_img = nib.Nifti1Image(combined_data, liver_img.affine, liver_img.header)
    combined_img.set_data_dtype(np.uint8)
    combined_img.get_data_dtype(finalize=True)
    nib.save(combined_img, combined_lesion_path)
    print(f"created combined lesion file: {combined_lesion_path}", flush=True)

def process_case(input_folder, case_id, classification_data, skip_existing):
    """
    Process a single case: create empty lesion masks and merge lesion files.
    """
    case_path = os.path.join(input_folder, case_id)
    segmentations_path = os.path.join(case_path, 'segmentations')
    lesion_path = os.path.join(segmentations_path, 'lesion.nii.gz')
    
    if not os.path.isdir(case_path):
        print(f"case directory not found: {case_id}, skipping", flush=True)
        return
    
    if skip_existing and os.path.isfile(lesion_path):
        print(f"skipping case {case_id} as lesion.nii.gz already exists.", flush=True)
        return
    
    liver_path = os.path.join(segmentations_path, 'liver.nii.gz')
    if not os.path.isfile(liver_path):
        print(f"liver file not found for case {case_id}, skipping", flush=True)
        return
    
    liver_img = nib.load(liver_path)
    if case_id in classification_data['BDMAP_ID'].values:
        classification_row = classification_data.loc[classification_data['BDMAP_ID'] == case_id].iloc[0]
        create_empty_lesion_masks(segmentations_path, liver_img, classification_row)

    lesion_files = get_lesion_files(segmentations_path)
    if lesion_files:
        merge_lesion_files(segmentations_path, liver_img, lesion_files)
    else:
        print(f"no lesion files found for case {case_id}, skipping lesion.nii.gz creation.", flush=True)

def process_abdomenatlas(input_folder, csv_file, ids_txt, max_workers, skip_existing):
    """
    Process selected cases in the AbdomenAtlas dataset.
    """
    classification_data = load_case_metadata(csv_file)
    selected_ids = load_ids_to_process(ids_txt)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_case, input_folder, case_id, classification_data, skip_existing): case_id
            for case_id in selected_ids
        }

        # use as_completed to update the progress bar properly
        with tqdm(total=len(futures), desc="Processing cases") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()  # retrieve results to catch any exceptions
                except Exception as e:
                    case_id = futures[future]
                    print(f"Error processing case {case_id}: {e}", flush=True)
                pbar.update(1)
def main():
    parser = argparse.ArgumentParser(description="process AbdomenAtlas cases.")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the AbdomenAtlasPre folder.")
    parser.add_argument("--classification_labels_csv", type=str, required=True, help="path to the classification labels csv file.")
    parser.add_argument("--process_ids_txt", type=str, required=True, help="path to the txt file with IDs to process.")
    parser.add_argument("--max_workers", type=int, default=os.cpu_count(), help="number of workers to use for multiprocessing.")
    parser.add_argument("--skip_existing", action="store_true", help="skip cases with existing lesion.nii.gz files.")
    args = parser.parse_args()
    
    process_abdomenatlas(args.dataset_path, args.classification_labels_csv, args.process_ids_txt, args.max_workers, args.skip_existing)

if __name__ == "__main__":
    main()
