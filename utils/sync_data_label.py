'''
python sync_data_label.py --source_datapath /mnt/T8/AbdomenAtlasPre/ --destination_datapath /ccvl/net/ccvl15/zzhou82/data/AbdomenAtlasPro/ --num_core 70 --mask

python sync_data_label.py --source_datapath /mnt/T8/AbdomenAtlasPre/ --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/ --num_core 76 --mask >logs/sync_mask.txt

python sync_data_label.py --source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_only/AbdomenAtlasPro/AbdomenAtlasPro/ --num_core 70 --ct >logs/sync_ct.txt

ccvl26 -> ccvl3:
python sync_data_label.py --source_datapath /mnt/T8/AbdomenAtlasPre/ --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlasProMini/AbdomenAtlasProMini/ --ct --mask --mini --num_core 76 >logs/sync_ct_mask_mini.txt

ccvl26 -> ccvl6:
python sync_data_label.py --source_datapath /mnt/T8/AbdomenAtlasPre/ --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlasPro/AbdomenAtlasPro/ --ct --mask --num_core 76 >logs/sync_ct_mask.txt

ccvl26 -> ccvl9:
python sync_data_label.py --source_datapath /mnt/T8/AbdomenAtlasPre/ --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_only/AbdomenAtlasProMini/AbdomenAtlasProMini/ --ct --mini --num_core 76 >logs/sync_ct_mini.txt

ccvl26 -> ccvl18:
python sync_data_label.py --source_datapath /mnt/T8/AbdomenAtlasPre/ --supp_source_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --destination_datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlasProMini/AbdomenAtlasProMini/ --mask --mini --num_core 76 >logs/sync_mask_mini.txt
'''

import os
import argparse
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from helper_functions import *

def event(pid, args):

    if args.ct:
        data, affine, header = load_ct(pid, args.supp_source_datapath)
        assert data is not None

        if np.min(data) < -1000 or np.max(data) > 1000:
            data[data<-1000] = -1000
            data[data>1000] = 1000
            print('>> min/max data error in {}, fixed'.format(pid))

        if args.mini:
            (min_x, min_y, min_z), (max_x, max_y, max_z) = crop_largest_subarray(arr=data, 
                                                                                 low_threshold=-100, 
                                                                                 high_threshold=100, 
                                                                                 )
            data = data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

        save_ct(data, affine, header, 
                pid, args.destination_datapath,
                )

    if args.mask:
        class_list = glob.glob(os.path.join(args.source_datapath, pid, 'segmentations', '*.nii.gz'))
        class_list = [c.split('/')[-1][:-7] for c in class_list]

        for class_name in class_list:
            
            # If '_' is the first letter of class_name and the class_name without '_' is in class_list, skip this class_name
            if class_name[0] == '_' and class_name[1:] in class_list:
                continue

            try:
                data, affine, header = load_mask(pid, class_name, args.source_datapath)
            except:
                print('>> cannot load mask: {} ({}), skipped'.format(pid, class_name))
                continue
            assert data is not None

            if np.min(data) < 0 or np.max(data) > 1:
                print('>> min/max mask error in {} ({}), fixed'.format(pid, class_name))
                data = data > 0.5

            if args.mini:

                # if min_x has not been defined, we need to load ct to get the min_x, min_y, min_z, max_x, max_y, max_z
                if 'min_x' not in locals():
                    ct_data, ct_affine, ct_header = load_ct(pid, args.supp_source_datapath)
                    assert ct_data is not None
                    if np.min(ct_data) < -1000 or np.max(ct_data) > 1000:
                        ct_data[ct_data<-1000] = -1000
                        ct_data[ct_data>1000] = 1000
                        print('>> min/max data error in {}, fixed'.format(pid))

                    (min_x, min_y, min_z), (max_x, max_y, max_z) = crop_largest_subarray(arr=ct_data, 
                                                                                         low_threshold=-100, 
                                                                                         high_threshold=100, 
                                                                                         )

                data = data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

            save_mask(data, affine, header, 
                      pid, class_name, args.destination_datapath,
                     )

def main(args):

    if args.destination_datapath is None:
        args.destination_datapath = args.source_datapath
    if not os.path.exists(args.destination_datapath):
        os.makedirs(args.destination_datapath)

    folder_names = [name for name in os.listdir(args.source_datapath) if os.path.isdir(os.path.join(args.source_datapath, name))]

    folder_names = sorted(folder_names)

    # TODO - begin delete 
    # folder_names = folder_names[:1000]
    # # print(folder_names)
    # for pid in tqdm(folder_names, ncols=80):
    #     event(pid, args)
    # TODO - end delete 

    if args.num_core > 0:
        num_core = args.num_core
    else:
        num_core = int(cpu_count())

    print('>> {} CPU cores are secured.'.format(num_core))
    
    with ProcessPoolExecutor(max_workers=num_core) as executor:

        futures = {executor.submit(event, pid, args): pid
                   for pid in folder_names}
        
        for future in tqdm(as_completed(futures), total=len(futures), ncols=80):
            folder = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {folder}: {e}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_datapath', dest='source_datapath', type=str, default='/Volumes/T9/AbdomenAtlasPro',
                        help='the directory of the source dataset',
                       )
    parser.add_argument('--supp_source_datapath', dest='supp_source_datapath', type=str, default='/Volumes/T9/AbdomenAtlasPro',
                        help='the directory of the supplementary AbdomenAtlas Pro dataset',
                       )
    parser.add_argument('--destination_datapath', dest='destination_datapath', type=str, default=None,
                        help='the directory of the destination dataset',
                       )
    parser.add_argument('--num_core', dest='num_core', type=int, default=0,
                        help='number of CPU core needed for this process',
                       )
    parser.add_argument('--ct', action='store_true', default=False, 
                        help='include ct?',
                       )
    parser.add_argument('--mask', action='store_true', default=False, 
                        help='include mask?',
                       )
    parser.add_argument('--mini', action='store_true', default=False, 
                        help='use mini version?',
                       )
    args = parser.parse_args()
    
    main(args)
