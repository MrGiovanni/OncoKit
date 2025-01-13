'''
python -W ignore create_normal_tumor_masks.py --datapath /mnt/T9/AbdomenAtlasPro --revison_file Tumor_Revision.csv
'''

import numpy as np
import os
import csv
import nibabel as nib
import argparse
from tqdm import tqdm

def save_mask(data, affine, header, pid, class_name, datapath):
    
    nifti_path = os.path.join(datapath, pid, 'segmentations', class_name + '.nii.gz')

    mask = nib.Nifti1Image(data, affine, header)
    mask.set_data_dtype(np.uint8)
    mask.get_data_dtype(finalize=True)
    nib.save(mask, nifti_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', dest='datapath', type=str, default='/Volumes/T9/AbdomenAtlasPro',
                        help='the directory of the AbdomenAtlas Pro dataset',
                        )
    parser.add_argument('--revison_file', type=str, default='Tumor_Revision.csv',
                        help='the path of the tumor revision file',
                        )
    args = parser.parse_args()

    tgt_root = args.datapath
    cases_info=[*csv.DictReader(open(args.revison_file))]

    for case in tqdm(cases_info):
        bdmap_name = case['BDMAP ID']

        if not os.path.exists(os.path.join(tgt_root, bdmap_name)):
            continue
        
        if os.path.exists(os.path.join(tgt_root, bdmap_name, 'segmentations')):
            img = nib.load(os.path.join(tgt_root, bdmap_name, 'segmentations/liver.nii.gz'))
        else:
            img = nib.load(os.path.join(tgt_root, bdmap_name, 'ct.nii.gz'))
        
        affine = img.affine
        header = img.header
        data = img.get_fdata()
        data.fill(0)

        if case['liver'] == 'N':
            save_mask(data=data, affine=affine, header=header, pid=bdmap_name, class_name='liver_tumor', datapath=tgt_root)

        if case['pancreas'] == 'N':
            save_mask(data=data, affine=affine, header=header, pid=bdmap_name, class_name='pancreatic_tumor', datapath=tgt_root)

        if case['kidney'] == 'N':
            save_mask(data=data, affine=affine, header=header, pid=bdmap_name, class_name='kidney_tumor', datapath=tgt_root)

if __name__ == "__main__":
    main()