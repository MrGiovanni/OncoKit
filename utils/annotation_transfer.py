'''
source /data/zzhou82/environments/syn/bin/activate
python -W ignore annotation_transfer.py --source_datapath /Volumes/T9/HGFC_inference_separate_data --destination_datapath /Volumes/T9/AbdomenAtlasPro -o error_analysis/aorta.csv -c aorta
python -W ignore annotation_transfer.py --source_datapath /Volumes/T9/HGFC_inference_separate_data --destination_datapath /Volumes/T9/AbdomenAtlasPro -o error_analysis/kidney.csv -c kidney_left
python -W ignore annotation_transfer.py --source_datapath /Volumes/Expansion/AbdomenAtlas/AbdomenAtlasPro --destination_datapath /Volumes/T9/AbdomenAtlasPro -o /Volumes/T9/error_analysis/aorta.csv -c aorta
python -W ignore annotation_transfer.py --source_datapath /Volumes/T9/AbdomenAtlas1.0 --destination_datapath /Volumes/T9/AbdomenAtlasPro -o case_id/AbdomenAtlas1.0.txt -c all
for id in BDMAP_00000003 BDMAP_00002407 BDMAP_00000219 BDMAP_00000320 BDMAP_00000233 BDMAP_00001230 BDMAP_00001636 BDMAP_00004457 BDMAP_00001396 BDMAP_00001483 BDMAP_00000406 BDMAP_00001045 BDMAP_00001647 BDMAP_00003411 BDMAP_00003976; do cp -r /Volumes/T9/AbdomenAtlasPro/$id /Users/zongwei.zhou/Dropbox\ \(ASU\)/PublicResource/Xinze/WeaklyRevision/; done
for class in colon bladder lung_left lung_right hepatic_vessel celiac_trunk intestine stomach duodenum rectum; do python -W ignore annotation_transfer.py --source_datapath /data2/wenxuan/Dataset/AbdomenAtlasManualRevision --destination_datapath /mnt/T9/AbdomenAtlasPro -o case_id/ManualRevisionWX.txt -c $class; done
python -W ignore annotation_transfer.py --source_datapath /mnt/ccvl15/xinze/weakly_revision --destination_datapath /mnt/T9/AbdomenAtlasPro -o case_id/ManualRevisionXZ.txt -c kidney_right
python -W ignore annotation_transfer.py --source_datapath /mnt/T8/AbdomenAtlasPre --destination_datapath /mnt/T9/AbdomenAtlasPro -c pro --num_core 78
'''

import os
import argparse
import csv
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from helper_functions import *

def event(pid, args):

    if args.class_name == 'all':

        for _, class_name in args.class_maps.items():
            shutil.copy(os.path.join(args.source_datapath, pid, 'segmentations', class_name+'.nii.gz'), 
                        os.path.join(args.destination_datapath, pid, 'segmentations', class_name+'.nii.gz'),
                        )
    elif args.class_name == 'pro':

        # copy whatever exists in the source, pid, segmentations folder to the destination, pid, segmentations folder. If destination, pid, segmentations does not exist, create it.
        if not os.path.exists(os.path.join(args.destination_datapath, pid, 'segmentations')):
            os.makedirs(os.path.join(args.destination_datapath, pid, 'segmentations'))

        # get all the class names in the source, pid, segmentations folder
        class_names = [f.split('.')[0] for f in os.listdir(os.path.join(args.source_datapath, pid, 'segmentations')) if f.endswith('.nii.gz')]
        for class_name in class_names:
            shutil.copy(os.path.join(args.source_datapath, pid, 'segmentations', class_name+'.nii.gz'), 
                        os.path.join(args.destination_datapath, pid, 'segmentations', class_name+'.nii.gz'),
                        )
    else:
        source_path = os.path.join(args.source_datapath, pid, 'segmentations', args.class_name + '.nii.gz')
        destination_path = os.path.join(args.destination_datapath, pid, 'segmentations', args.class_name + '.nii.gz')
        if os.path.isfile(source_path):

            if get_dim(os.path.join(args.destination_datapath, pid, 'ct.nii.gz')) == get_dim(source_path):
                shutil.copy(source_path, destination_path)
                if args.class_name == 'aorta':
                    if aorta_error(pid, args.destination_datapath):
                        print('Success: {}, but the error remains!'.format(destination_path))
                    else:
                        print('Success: {}'.format(destination_path))
                if 'kidney' in args.class_name:
                    if kidney_error(pid, args.destination_datapath):
                        print('Success: {}, but the error remains!'.format(destination_path))
                    else:
                        print('Success: {}'.format(destination_path))
            else:
                print('Error: {} shape mismatch'.format(pid))
        else:
            print('>> Error: no such file {}'.format(source_path))

def main(args):

    if args.version == '1.0':
        args.class_maps = class_map_abdomenatlas_1_0
    elif args.version == '1.1':
        args.class_maps = class_map_abdomenatlas_1_1
    else:
        raise

    replace_id_list = []

    if args.class_name == 'pro':
        # get all the case ids in the source_datapath
        replace_id_list = [f for f in os.listdir(args.source_datapath) if os.path.isdir(os.path.join(args.source_datapath, f))]
        replace_id_list = sorted(replace_id_list)
    else:
        with open(args.csvpath, newline='') as file:
            reader = csv.reader(file)
            if '.csv' in args.csvpath:
                next(reader)
            for row in reader:
                replace_id_list.append(row[0])

    print('A total of {} subjects need to be replaced.'.format(len(replace_id_list)))
    
    # multi-core processing
    if args.num_core > 0:
        num_core = args.num_core
    else:
        num_core = int(cpu_count())

    print('>> {} CPU cores are secured.'.format(num_core))
    
    with ProcessPoolExecutor(max_workers=num_core) as executor:
        futures = {executor.submit(event, pid, args): pid
                   for pid in replace_id_list}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            folder = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {folder}: {e}")
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_datapath', dest='source_datapath', type=str, default='/Volumes/T9/HGFC_inference_separate_data',
                        help='the directory of the AbdomenAtlas dataset',
                       )
    parser.add_argument('--destination_datapath', dest='destination_datapath', type=str, default='/Volumes/T9/AbdomenAtlasPro',
                        help='the directory of the AbdomenAtlas dataset',
                       )
    parser.add_argument('--num_core', dest='num_core', type=int, default=0,
                        help='the number of CPU cores',
                        )
    parser.add_argument('-o', dest='csvpath', type=str, default='error_analysis/aorta.csv',
                        help='the directory of the saved csv file recording all the error cases',
                       )
    parser.add_argument('-c', dest='class_name', type=str, default='aorta',
                        help='the class name in error cases',
                       )
    parser.add_argument('-v', dest='version', type=str, default='1.0',
                        help='the version of dataset',
                       )
    args = parser.parse_args()
    
    main(args)