'''
source /data/zzhou82/environments/syn/bin/activate
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /Volumes/T9/AbdomenAtlas1.0 --ct --mask -v 1.0
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /Volumes/T9/AbdomenAtlas1.1 --ct --mask -v 1.1
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /Volumes/T9/AbdomenAtlas1.0Mini --ct --mask -v 1.0mini

# 1.0
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas1.0 --ct --mask -v 1.0
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/ccvl15/zzhou82/data/AbdomenAtlas/image_only/AbdomenAtlas1.0 --ct -v 1.0
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/ccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlas1.0 --mask -v 1.0
for id in BDMAP_00000339 BDMAP_00001024 BDMAP_00001044 BDMAP_00001230 BDMAP_00001483 BDMAP_00001647 BDMAP_00001992 BDMAP_00002727 BDMAP_00003411 BDMAP_00003888 BDMAP_00004457 BDMAP_00000419 BDMAP_00001032 BDMAP_00001045 BDMAP_00001396 BDMAP_00001636 BDMAP_00001882 BDMAP_00002407 BDMAP_00003292 BDMAP_00003725 BDMAP_00003976 BDMAP_00005047; do echo $id; python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /ccvl/net/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas1.0 --ct --mask -v 1.0 --patientID $id; done

# 1.0Mini
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas1.0Mini --ct --mask -v 1.0mini
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/ccvl15/zzhou82/data/AbdomenAtlas/image_only/AbdomenAtlas1.0Mini --ct -v 1.0mini
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/ccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlas1.0Mini --mask -v 1.0mini
for id in BDMAP_00000339 BDMAP_00001024 BDMAP_00001044 BDMAP_00001230 BDMAP_00001483 BDMAP_00001647 BDMAP_00001992 BDMAP_00002727 BDMAP_00003411 BDMAP_00003888 BDMAP_00004457 BDMAP_00000419 BDMAP_00001032 BDMAP_00001045 BDMAP_00001396 BDMAP_00001636 BDMAP_00001882 BDMAP_00002407 BDMAP_00003292 BDMAP_00003725 BDMAP_00003976 BDMAP_00005047; do echo $id; python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /ccvl/net/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas1.0Mini --ct --mask -v 1.0mini --patientID $id; done

# 1.1
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas1.1 --ct --mask -v 1.1
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/ccvl15/zzhou82/data/AbdomenAtlas/image_only/AbdomenAtlas1.1 --ct -v 1.1
python -W ignore generate_versions.py --source_datapath /ccvl/net/ccvl15/zzhou82/data/AbdomenAtlasPro --destination_datapath /ccvl/net/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas1.1/AbdomenAtlas1.1 --mask -v 1.1
for id in BDMAP_00000339 BDMAP_00001024 BDMAP_00001044 BDMAP_00001230 BDMAP_00001483 BDMAP_00001647 BDMAP_00001992 BDMAP_00002727 BDMAP_00003411 BDMAP_00003888 BDMAP_00004457 BDMAP_00000419 BDMAP_00001032 BDMAP_00001045 BDMAP_00001396 BDMAP_00001636 BDMAP_00001882 BDMAP_00002407 BDMAP_00003292 BDMAP_00003725 BDMAP_00003976 BDMAP_00005047; do echo $id; python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /ccvl/net/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas1.1 --ct --mask -v 1.1 --patientID $id; done

# 1.1Mini
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas1.1Mini --ct --mask -v 1.1mini
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/ccvl15/zzhou82/data/AbdomenAtlas/image_only/AbdomenAtlas1.1Mini --ct -v 1.1mini
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/ccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlas1.1Mini --mask -v 1.1mini
for id in BDMAP_00000339 BDMAP_00001024 BDMAP_00001044 BDMAP_00001230 BDMAP_00001483 BDMAP_00001647 BDMAP_00001992 BDMAP_00002727 BDMAP_00003411 BDMAP_00003888 BDMAP_00004457 BDMAP_00000419 BDMAP_00001032 BDMAP_00001045 BDMAP_00001396 BDMAP_00001636 BDMAP_00001882 BDMAP_00002407 BDMAP_00003292 BDMAP_00003725 BDMAP_00003976 BDMAP_00005047; do echo $id; python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /ccvl/net/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas1.1Mini --ct --mask -v 1.1mini --patientID $id; done

# 2.0
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas2.0/AbdomenAtlas2.0 --ct --mask -v 2.0
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_only/AbdomenAtlas2.0/AbdomenAtlas2.0 --ct -v 2.0
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlas2.0/AbdomenAtlas2.0 --mask -v 2.0
for id in BDMAP_00001353 BDMAP_00001358; do echo $id; python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlas2.0/AbdomenAtlas2.0 --mask -v 2.0 --patientID $id; done

# 2.0Mini
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas2.0Mini/AbdomenAtlas2.0Mini --ct --mask -v 2.0mini
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_only/AbdomenAtlas2.0Mini/AbdomenAtlas2.0Mini --ct -v 2.0mini
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlas2.0Mini/AbdomenAtlas2.0Mini --mask -v 2.0mini
for id in BDMAP_00001353 BDMAP_00001358; do echo $id; python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlas2.0Mini/AbdomenAtlas2.0Mini --mask -v 2.0mini --patientID $id; done

# 3.0
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas3.0/AbdomenAtlas3.0 --ct --mask -v 3.0
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_only/AbdomenAtlas3.0/AbdomenAtlas3.0 --ct -v 3.0
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlas3.0/AbdomenAtlas3.0 --mask -v 3.0
for id in BDMAP_00001353 BDMAP_00001358; do echo $id; python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlas3.0/AbdomenAtlas3.0 --mask -v 3.0 --patientID $id; done

# 3.0Mini
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas3.0Mini/AbdomenAtlas3.0Mini --ct --mask -v 3.0mini
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_only/AbdomenAtlas3.0Mini/AbdomenAtlas3.0Mini --ct -v 3.0mini
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlas3.0Mini/AbdomenAtlas3.0Mini --mask -v 3.0mini
for id in BDMAP_00001353 BDMAP_00001358; do echo $id; python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlas3.0Mini/AbdomenAtlas3.0Mini --mask -v 3.0mini --patientID $id; done

# X
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/imgae_mask/AbdomenAtlasX/AbdomenAtlasX --ct --mask -v X --num_core 60
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlasX/AbdomenAtlasX --mask -v X --num_core 60
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/imgae_mask/AbdomenAtlasX/AbdomenAtlasX --ct --mask -v X --num_core 2 --patientID BDMAP_00002098

# XMini
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/imgae_mask/AbdomenAtlasXMini/AbdomenAtlasXMini --ct --mask -v Xmini --num_core 60
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlasXMini/AbdomenAtlasXMini --mask -v Xmini --num_core 60

# imagecas
python -W ignore generate_versions.py --source_datapath /mnt/T8/AbdomenAtlasPre --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /data/zzhou82/data/ImageCAS --ct --mask -v imagecas

# report
python -W ignore generate_versions.py --source_datapath ../AbdomenAtlasReport --destination_datapath ./Report --ct --mask -v report

# All in one
python -W ignore generate_versions.py --source_datapath /mnt/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas1.0Mini --destination_datapath /mnt/ccvl15/zzhou82/data/AbdomenAtlas/image_mask --tar_gz_name AbdomenAtlas1.0Mini --all_in_one

# Create multiple zips for single patients
python -W ignore generate_versions.py --source_datapath /mnt/ccvl15/zzhou82/data/TEST/AbdomenAtlasX --destination_datapath /mnt/ccvl15/zzhou82/data/TEST --zips_folder_name _AbdomenAtlasX --zips

# Create multiple zips for multiple patients (by default, every 500)
version=AbdomenAtlas1.1
python -W ignore generate_versions.py --source_datapath /ccvl/net/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/$version/$version --destination_datapath /ccvl/net/ccvl15/zzhou82/data/AbdomenAtlas/_image_mask/_$version/_$version --zips_folder_name $version --many_in_one
'''

import os
import argparse
import shutil
import csv
import glob
import tarfile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from helper_functions import *

def make_tarfile(output_tar, folder_path):
    # Create a tar.gz file
    with tarfile.open(output_tar, 'w:gz') as tar:
        # Iterate through all files and directories within the specified folder
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            # Add each item to the tar file, without including the parent folder
            tar.add(item_path, arcname=item)

def event(pid, args):

    if not os.path.exists(os.path.join(args.source_datapath, pid)):
        print('>> {} does not exist.'.format(pid))
        return

    if 'mini' in args.version.lower():

        # Load ct
        if not os.path.exists(os.path.join(args.destination_datapath, pid)):
            os.makedirs(os.path.join(args.destination_datapath, pid))

        # copy ct
        # if os.path.join(args.source_datapath, pid, 'ct.nii.gz') does not exist
        if os.path.isfile(os.path.join(args.source_datapath, pid, 'ct.nii.gz')):
            shutil.copy(os.path.join(args.source_datapath, pid, 'ct.nii.gz'), 
                        os.path.join(args.destination_datapath, pid, 'ct.nii.gz'),
                        )
        else:
            shutil.copy(os.path.join(args.supp_source_datapath, pid, 'ct.nii.gz'), 
                        os.path.join(args.destination_datapath, pid, 'ct.nii.gz'),
                        )
        
        # Load mask
        if not os.path.exists(os.path.join(args.destination_datapath, pid, 'segmentations')):
            os.makedirs(os.path.join(args.destination_datapath, pid, 'segmentations'))
        # copy segmentation masks
        for cid, class_name in args.class_maps.items():
            if os.path.isfile(os.path.join(args.source_datapath, pid, 'segmentations', class_name+'.nii.gz')):
                shutil.copy(os.path.join(args.source_datapath, pid, 'segmentations', class_name+'.nii.gz'), 
                            os.path.join(args.destination_datapath, pid, 'segmentations', class_name+'.nii.gz'),
                            )
            
        # Save combined labels
        combined_labels, affine, header = load_mask(pid, args.class_maps[1], args.destination_datapath)
        combined_labels.fill(0)
        for cid, class_name in args.class_maps.items():
            mask, _, _ = load_mask(pid, class_name, args.destination_datapath)
            if mask is not None:
                combined_labels[mask > 0.5] = cid
        nifti_path = os.path.join(args.destination_datapath, pid, 'combined_labels.nii.gz')
        nib.save(nib.Nifti1Image(combined_labels.astype(np.int8), affine=affine, header=header), nifti_path)

        # Create mini version
        mini_ct_file = os.path.join(args.destination_datapath, pid, 'ct.nii.gz')

        mini_comb_file = os.path.join(args.destination_datapath, pid, 'combined_labels.nii.gz')
        mini_mask_file = glob.glob(os.path.join(args.destination_datapath, pid, 'segmentations/*.nii.gz'))
        standardization(original_ct_file=mini_ct_file, revised_ct_file=mini_ct_file,
                        original_mask_file=mini_mask_file, revised_mask_file=mini_mask_file,
                        original_comb_file=mini_comb_file, revised_comb_file=mini_comb_file,
                        image_type=np.int16, mask_type=np.int8,
                        )

        if not args.ct:
            os.remove(mini_ct_file)
        if not args.mask:
            os.remove(mini_comb_file)
            shutil.rmtree(os.path.join(args.destination_datapath, pid, 'segmentations'))

    else:
        if args.ct:
            if not os.path.exists(os.path.join(args.destination_datapath, pid)):
                os.makedirs(os.path.join(args.destination_datapath, pid))

            # copy ct
            # if os.path.join(args.source_datapath, pid, 'ct.nii.gz') does not exist
            if os.path.isfile(os.path.join(args.source_datapath, pid, 'ct.nii.gz')):
                shutil.copy(os.path.join(args.source_datapath, pid, 'ct.nii.gz'), 
                            os.path.join(args.destination_datapath, pid, 'ct.nii.gz'),
                            )
            else:
                shutil.copy(os.path.join(args.supp_source_datapath, pid, 'ct.nii.gz'), 
                            os.path.join(args.destination_datapath, pid, 'ct.nii.gz'),
                            )
        
        if args.mask:
            if not os.path.exists(os.path.join(args.destination_datapath, pid, 'segmentations')):
                os.makedirs(os.path.join(args.destination_datapath, pid, 'segmentations'))
            # copy segmentation masks
            for cid, class_name in args.class_maps.items():
                if os.path.isfile(os.path.join(args.source_datapath, pid, 'segmentations', class_name+'.nii.gz')):
                    shutil.copy(os.path.join(args.source_datapath, pid, 'segmentations', class_name+'.nii.gz'), 
                                os.path.join(args.destination_datapath, pid, 'segmentations', class_name+'.nii.gz'),
                                )
            
            # Save combined labels
            combined_labels, affine, header = load_mask(pid, args.class_maps[1], args.destination_datapath)
            combined_labels.fill(0)
            for cid, class_name in args.class_maps.items():
                mask, _, _ = load_mask(pid, class_name, args.destination_datapath)
                if mask is not None:
                    combined_labels[mask > 0.5] = cid
            nifti_path = os.path.join(args.destination_datapath, pid, 'combined_labels.nii.gz')
            nib.save(nib.Nifti1Image(combined_labels.astype(np.int8), affine=affine, header=header), nifti_path)

def event_compression(pid, args):

    # copy folder from source to destination.
    shutil.copytree(os.path.join(args.source_datapath, pid), 
                    os.path.join(args.destination_datapath, args.zips_folder_name, pid),
                    )
    # make tar file for this folder.
    make_tarfile(os.path.join(args.destination_datapath, args.zips_folder_name, pid+'.tar.gz'), 
                    os.path.join(args.destination_datapath, args.zips_folder_name, pid),
                    )

    # delete this folder.
    shutil.rmtree(os.path.join(args.destination_datapath, args.zips_folder_name, pid))

def generate_new_version(args):

    print('>> Generating {} version'.format(args.version))

    if '1.0' in args.version:
        args.class_maps = class_map_abdomenatlas_1_0
        args.id_txt = '1.0'
    elif '1.1' in args.version:
        args.class_maps = class_map_abdomenatlas_1_1
        args.id_txt = '1.1'
    elif '1.2' in args.version:
        args.class_maps = class_map_abdomenatlas_1_2
        args.id_txt = '1.2'
    elif '2.0' in args.version:
        args.class_maps = class_map_abdomenatlas_2_0
        args.id_txt = '2.0'
    elif '3.0' in args.version:
        args.class_maps = class_map_abdomenatlas_3_0
        args.id_txt = '3.0'
    elif 'X' in args.version:
        args.class_maps = class_map_abdomenatlas_x
        args.id_txt = '2.0'
    elif 'imagecas' in args.version:
        args.class_maps = class_map_abdomenatlas_imagecas
        args.id_txt = 'ImageCAS'
    elif 'report' in args.version:
        args.class_maps = class_map_abdomenatlas_report
        args.id_txt = 'Report'
    else:
        raise

    id_list = []

    with open(os.path.join('case_id', 'AbdomenAtlas'+args.id_txt+'.txt'), newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            id_list.append(row[0])

    if not os.path.exists(args.destination_datapath):
        os.makedirs(args.destination_datapath)

    if args.patientID is None:
        if args.num_core > 0:
            num_core = args.num_core
        else:
            num_core = int(cpu_count())

        print('>> {} CPU cores are secured.'.format(num_core))
        
        with ProcessPoolExecutor(max_workers=num_core) as executor:

            futures = {executor.submit(event, pid, args): pid
                    for pid in id_list}
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                folder = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {folder}: {e}")

    else:
        event(args.patientID, args)

def compress_existing_version(args):

    if not os.path.exists(args.destination_datapath):
        os.makedirs(args.destination_datapath)

    if args.zips:
        if not os.path.exists(os.path.join(args.destination_datapath, args.zips_folder_name)):
            os.makedirs(os.path.join(args.destination_datapath, args.zips_folder_name))
        
        folder_names = [name for name in os.listdir(args.source_datapath) if os.path.isdir(os.path.join(args.source_datapath, name))]
        folder_names = sorted(folder_names)
        print('>> Save to {} zips'.format(len(folder_names)))

        if args.num_core > 0:
            num_core = args.num_core
        else:
            num_core = int(cpu_count())

        print('>> {} CPU cores are secured.'.format(num_core))

        with ProcessPoolExecutor(max_workers=num_core) as executor:

            futures = {executor.submit(event_compression, pid, args): pid
                    for pid in folder_names}
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                folder = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {folder}: {e}")

    if args.all_in_one:
        print('>> Save to one tar.gz')
        make_tarfile(os.path.join(args.destination_datapath, args.tar_gz_name+'.tar.gz'), args.source_datapath)
    
    if args.many_in_one:
        # save a zip for every num_patient patients.
        # for example, if num_patient = 500, then the name of each zip is BDMAP_00000001_00000500.tar.gz, BDMAP_00000501_00001000.tar.gz, and so on.
        num_patient = 500
        folder_names = [name for name in os.listdir(args.source_datapath) if os.path.isdir(os.path.join(args.source_datapath, name))]
        folder_names = sorted(folder_names)
        print('>> Save to multiple zips')
        for i in tqdm(range(0, len(folder_names), num_patient)):
            if not os.path.exists(os.path.join(args.destination_datapath, args.zips_folder_name+str(i))):
                os.makedirs(os.path.join(args.destination_datapath, args.zips_folder_name+str(i)))
            for j in range(i, min(i+num_patient, len(folder_names))):
                shutil.copytree(os.path.join(args.source_datapath, folder_names[j]), 
                                os.path.join(args.destination_datapath, args.zips_folder_name+str(i), folder_names[j]),
                                )
            # make tar file for every thing in this folder, not this folder itself.
            make_tarfile(output_tar=os.path.join(args.destination_datapath, 
                                                 args.zips_folder_name+'_BDMAP_'+str(i+1).zfill(8)+'_'+str(min(i+num_patient, len(folder_names))).zfill(8)+'.tar.gz'), 
                         folder_path=os.path.join(args.destination_datapath, args.zips_folder_name+str(i)),
                        )
            shutil.rmtree(os.path.join(args.destination_datapath, args.zips_folder_name+str(i)))

def main(args):

    if args.ct or args.mask:
        generate_new_version(args)
    
    if args.zips or args.all_in_one or args.many_in_one:
        compress_existing_version(args)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_datapath', dest='source_datapath', type=str, default='/Volumes/T9/AbdomenAtlasPro',
                        help='the directory of the AbdomenAtlas Pro dataset',
                       )
    parser.add_argument('--supp_source_datapath', dest='supp_source_datapath', type=str, default='/Volumes/T9/AbdomenAtlasPro',
                        help='the directory of the supplementary AbdomenAtlas Pro dataset',
                       )
    parser.add_argument('--destination_datapath', dest='destination_datapath', type=str, default='/Volumes/T9/AbdomenAtlasPro',
                        help='the directory of the AbdomenAtlas V_ dataset',
                       )
    parser.add_argument('--num_core', dest='num_core', type=int, default=0,
                        help='number of CPU core needed for this process',
                       )
    parser.add_argument('--patientID', dest='patientID', type=str, default=None,
                        help='patient ID to be generated',
                       )
    parser.add_argument('-v', dest='version', type=str, default='1.0',
                        help='the version of dataset',
                       )
    parser.add_argument('--ct', action='store_true', default=False, 
                        help='include ct?',
                       )
    parser.add_argument('--mask', action='store_true', default=False, 
                        help='include mask?',
                       )
    parser.add_argument('--zips', action='store_true', default=False, 
                        help='create multiple zips for the dataset?',
                       )
    parser.add_argument('--zips_folder_name', dest='zips_folder_name', type=str, default=None,
                        help='the name of zips folder',
                       )
    parser.add_argument('--all_in_one', action='store_true', default=False, 
                        help='create a sinlge zip for the dataset?',
                       )
    parser.add_argument('--many_in_one', action='store_true', default=False, 
                        help='create multiple zips for mutliple patients (by default, every 500)?',
                       )
    parser.add_argument('--tar_gz_name', dest='tar_gz_name', type=str, default=None,
                        help='the name of all-in-one tar file',
                       )
    
    args = parser.parse_args()
    
    main(args)
