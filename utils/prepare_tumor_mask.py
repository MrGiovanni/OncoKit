'''
python -W ignore prepare_tumor_mask.py --source_datapath /mnt/T8/AbdomenAtlasPre --num_core 70 >> logs/tumor_analysis_V3.txt

1. Rename the tumor mask files.
2. Do post-processing to remove small connected components, tumors outside the organ.
3. Potential error detetion.
'''

import os
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from helper_functions import *

import numpy as np
from scipy.ndimage import label
from scipy.ndimage import binary_dilation

# Create a 3D spherical structuring element with a radius of 5 pixels
def create_structuring_element(radius):
    grid = np.ogrid[
        -radius:radius+1, 
        -radius:radius+1, 
        -radius:radius+1
    ]
    struct_elem = (grid[0]**2 + grid[1]**2 + grid[2]**2) <= radius**2
    return struct_elem

def tumor_postprocessing(tumor_mask, organ_mask):

    """
    Remove tumor components from the tumor mask that do not overlap with the organ mask.

    Args:
        tumor_mask (numpy.ndarray): 3D binary mask of the tumor (1 for tumor, 0 otherwise).
        organ_mask (numpy.ndarray): 3D binary mask of the organ (1 for organ, 0 otherwise).

    Returns:
        numpy.ndarray: Updated tumor mask with non-overlapping components removed.
    """

    tumor_mask = tumor_mask > 0.5

    # Label connected components in the tumor mask
    labeled_tumors, num_tumors = label(tumor_mask)

    # Initialize an empty mask for the result
    filtered_tumor_mask = np.zeros_like(tumor_mask, dtype=np.uint8)

    # dilate organ_mask to 5 pixel in 3D space
    structuring_element = create_structuring_element(1)

    # Apply the binary dilation
    dilated_organ_mask = binary_dilation(organ_mask, structure=structuring_element)

    for tumor_label in range(1, num_tumors + 1):
        # Extract the current tumor component
        tumor_component = (labeled_tumors == tumor_label)

        # Check for overlap with the organ mask
        if np.any(tumor_component & dilated_organ_mask):
            # If there is overlap, retain this component in the result
            filtered_tumor_mask[tumor_component] = 1

    return filtered_tumor_mask > 0.5

def revise_tumor_mask(pid, tumor_name, organ_mask, args):

    if ((tumor_name == 'kidney_cyst' or tumor_name == 'kidney_tumor')) and '_A' not in pid and '_V' not in pid and '_O' not in pid: # do not modify ground truth in KiTS
        if int(pid.split('_')[-1]) <= 5195:
            return

    tumor_mask, tumor_affine, tumor_header = load_mask(pid, tumor_name, args.source_datapath)
    if tumor_mask is not None:
        filtered_tumor_mask = tumor_postprocessing(tumor_mask, organ_mask)
        if np.sum(filtered_tumor_mask - tumor_mask) != 0:
            save_mask(filtered_tumor_mask, tumor_affine, tumor_header, 
                    pid, tumor_name, args.destination_datapath,
                    )
            print(f'{tumor_name} optimized in {pid}')

def event(pid, args):

    # if a folder does not exist, create it
    if not os.path.exists(os.path.join(args.destination_datapath, pid, 'segmentations')):
        os.makedirs(os.path.join(args.destination_datapath, pid, 'segmentations'))
        print(f'Created folder {os.path.join(args.destination_datapath, pid, "segmentations")}')

    rename_delete_tumor_mask(pid, args)

    class_list = glob.glob(os.path.join(args.source_datapath, pid, 'segmentations', '*.nii.gz'))
    class_list = [c.split('/')[-1][:-7] for c in class_list]
    if 'liver_tumor' in class_list:
        print(f'>> {pid} liver_tumor exists')
    if 'pancreatic_tumor' in class_list:
        print(f'>> {pid} pancreatic_tumor exists')
    if 'kidney_tumor' in class_list and int(pid.split('_')[-1]) > 5195:
        print(f'>> {pid} kidney_tumor exists')

    kidney_left, kidney_left_affine, kidney_left_header = load_mask(pid, 'kidney_left', args.source_datapath)
    kidney_right, kidney_right_affine, kidney_right_header = load_mask(pid, 'kidney_right', args.source_datapath)
    if kidney_left is not None and kidney_right is not None:
        # merge kidney_left and kidney_right into kidney binary mask
        kidney = (kidney_left + kidney_right) > 0.5
        revise_tumor_mask(pid, tumor_name='kidney_cyst', organ_mask=kidney, args=args)
        revise_tumor_mask(pid, tumor_name='kidney_tumor', organ_mask=kidney, args=args)
        revise_tumor_mask(pid, tumor_name='kidney_lesion', organ_mask=kidney, args=args)
        revise_tumor_mask(pid, tumor_name='_kidney_lesion', organ_mask=kidney, args=args)
    
    liver, liver_affine, liver_header = load_mask(pid, 'liver', args.source_datapath)
    if liver is not None:
        revise_tumor_mask(pid, tumor_name='liver_lesion', organ_mask=liver, args=args)
        revise_tumor_mask(pid, tumor_name='_liver_lesion', organ_mask=liver, args=args)

    pancreas, pancreas_affine, pancreas_header = load_mask(pid, 'pancreas', args.source_datapath)
    if pancreas is not None:
        revise_tumor_mask(pid, tumor_name='pancreatic_lesion', organ_mask=pancreas, args=args)
        revise_tumor_mask(pid, tumor_name='pancreatic_tumor', organ_mask=pancreas, args=args)
        revise_tumor_mask(pid, tumor_name='pancreatic_cyst', organ_mask=pancreas, args=args)
        revise_tumor_mask(pid, tumor_name='pancreatic_pdac', organ_mask=pancreas, args=args)
        revise_tumor_mask(pid, tumor_name='pancreatic_pnet', organ_mask=pancreas, args=args)
        revise_tumor_mask(pid, tumor_name='_pancreatic_lesion', organ_mask=pancreas, args=args)

def main(args):

    if args.destination_datapath is None:
        args.destination_datapath = args.source_datapath
    if not os.path.exists(args.destination_datapath):
        os.makedirs(args.destination_datapath)

    folder_names = [name for name in os.listdir(args.source_datapath) if os.path.isdir(os.path.join(args.source_datapath, name))]
    folder_names = sorted(folder_names)

    if args.num_core > 0:
        num_core = args.num_core
    else:
        num_core = int(cpu_count())

    print('>> using {} cores'.format(num_core))

    # # get folder_names every 100 folders
    # folder_names = folder_names[::100]
    # for pid in tqdm(folder_names, desc='Processing', ncols=80):
    #     event(pid, args)

    with ProcessPoolExecutor(max_workers=num_core) as executor:

        futures = {executor.submit(event, pid, args): pid for pid in folder_names}

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing', ncols=80):

            folder = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f'Error processing {folder}: {exc}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare tumor mask')
    parser.add_argument('--source_datapath', type=str, required=True, help='source datapath')
    parser.add_argument('--destination_datapath', type=str, required=False, help='destination datapath')
    parser.add_argument('--num_core', type=int, default=0, help='number of cores')

    args = parser.parse_args()
    
    main(args)