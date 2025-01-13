'''
python -W ignore add_new_annotations.py --source_datapath /data2/wenxuan/Benchmark/STUNet/STU-Net/AbdomenAtlas/AbdomenAtlasProResults_base_ep2k --destination_datapath /mnt/T8/AbdomenAtlasPre --num_core 40
python -W ignore add_new_annotations.py --source_datapath /data2/wenxuan/code/benchmark/vista3d_inference/eval --destination_datapath /mnt/T8/AbdomenAtlasPre --num_core 70

python -W ignore add_new_annotations.py --source_datapath /ccvl/net/ccvl15/pedro/AtlasVessels/ --destination_datapath /mnt/T8/AbdomenAtlasPre --source_folder segmentations --num_core 66
python -W ignore add_new_annotations.py --source_datapath /ccvl/net/ccvl15/pedro/AtlasVessels/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --source_folder segmentations --num_core 66

python -W ignore add_new_annotations.py --source_datapath /ccvl/net/ccvl15/pedro/PancreasSegmentsAtlas --destination_datapath /mnt/T8/AbdomenAtlasPre --source_folder segmentations --num_core 66
python -W ignore add_new_annotations.py --source_datapath /ccvl/net/ccvl15/pedro/PancreasSegmentsAtlas --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --source_folder segmentations --num_core 66
'''

import os
import argparse
import shutil
from tqdm import tqdm
from helper_functions import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# TODO Complete this function
def label_expert(ct_path, mask1_path, mask2_path, class_name):

    mask1 = nib.load(mask1_path).get_fdata().astype(np.uint8)
    mask2 = nib.load(mask2_path).get_fdata().astype(np.uint8)
    ct = nib.load(ct_path).get_fdata().astype(np.int16)

    if class_name in class_map_abdomenatlas_1_0.values():
        better_path = label_critic(ct, mask1, mask2, class_name)
    else:
        better_path = medical_prior(ct, mask1, mask2, class_name)
    
    return better_path

def event(pid, args):

    # if segmentation file exists in source path and it is not in the destination path, copy it to the destination path.

    # get all the class names in the source path, pid, predictions folder
    class_names = os.listdir(os.path.join(args.source_datapath, pid, args.source_folder))
    for class_name in class_names:
        source_path = os.path.join(args.source_datapath, pid, args.source_folder, class_name)
        destination_path = os.path.join(args.destination_datapath, pid, 'segmentations', class_name)

        # if destination, pid, segmentations folder does not exist, create it
        if not os.path.exists(os.path.join(args.destination_datapath, pid, 'segmentations')):
            os.makedirs(os.path.join(args.destination_datapath, pid, 'segmentations'))

        if not os.path.isfile(destination_path):
            shutil.copy(source_path, destination_path)
            # print('Success: {}, added'.format(destination_path))
        else:
            # print('Error: {} already exists'.format(destination_path))
            # TODO: label expert
            # better_path = label_expert(ct_path, source_path, destination_path, class_name)
            # shutil.copy(better_path, destination_path)
            pass
           
def main(args):

    # get all the folder names in the source path
    source_pids = os.listdir(args.source_datapath)
    destination_pids = os.listdir(args.destination_datapath)

    # get the overlapping folder names
    pids = set(source_pids) & set(destination_pids)

    # multi-core processing
    if args.num_core > 0:
        num_core = args.num_core
    else:
        num_core = int(cpu_count())
    
    print('>> {} CPU cores are secured.'.format(num_core))
    with ProcessPoolExecutor(max_workers=num_core) as executor:
        futures = [executor.submit(event, pid, args) for pid in pids]
        for future in tqdm(as_completed(futures), total=len(futures)):
            folder = future.result()
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {folder}: {e}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Add new annotations to the destination path')
    parser.add_argument('--source_datapath', type=str, help='Path to the source data')
    parser.add_argument('--destination_datapath', type=str, help='Path to the destination data')
    parser.add_argument('--source_folder', type=str, default='predictions', help='Source folder name (e.g., predictions, segmentations)')
    parser.add_argument('--num_core', dest='num_core', type=int, default=0,
                        help='number of CPU core needed for this process',
                       )
    args = parser.parse_args()

    main(args)