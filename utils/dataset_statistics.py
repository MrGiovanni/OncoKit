'''
source /data/zzhou82/environments/syn/bin/activate
python -W ignore dataset_statistics.py --datapath /Users/zongwei.zhou/Desktop/VERSION_CHECK/1.1Mini -o /Users/zongwei.zhou/Desktop/dataset_statistics --num_slices --csvname z_slices
python -W ignore dataset_statistics.py --datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ -o /mnt/T8/error_analysis/ --num_slices --csvname z_slices --num_core 2
'''

import os
import argparse
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from helper_functions import *

def event(pid, args):

    num_slices = count_num_slices(pid, args.datapath)

    with open(os.path.join(args.csvpath, args.csvname+'.csv'), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([pid, num_slices])

def main(args):

    if os.path.isfile(os.path.join(args.csvpath, args.csvname+'.csv')):
        os.remove(os.path.join(args.csvpath, args.csvname+'.csv'))
    if not os.path.exists(args.csvpath):
        os.makedirs(args.csvpath)
    with open(os.path.join(args.csvpath, args.csvname+'.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Patient ID', 'Number of Slices'])

    folder_names = [name for name in os.listdir(args.datapath) if os.path.isdir(os.path.join(args.datapath, name))]
    # folder_names = sorted(folder_names)
    # total_slices = 0

    if args.num_core > 0:
        num_core = args.num_core
    else:
        num_core = int(cpu_count())

    print('>> {} CPU cores are secured.'.format(num_core))

    with ProcessPoolExecutor(max_workers=num_core) as executor:

        futures = {executor.submit(event, pid, args): pid for pid in folder_names}

        for future in tqdm(as_completed(futures), total=len(futures), ncols=80):
            folder = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {folder}: {e}")
    
    # for pid in tqdm(folder_names):

    #     num_slices = count_num_slices(pid, args.datapath)
    #     total_slices += num_slices

    #     with open(os.path.join(args.csvpath, args.csvname+'.csv'), 'a', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow([pid, num_slices])

    # print('There are a total of {} CT slices in {}'.format(total_slices, args.datapath))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', dest='datapath', type=str, default='/Volumes/T9/AbdomenAtlasPro',
                        help='the directory of the AbdomenAtlas dataset',
                       )
    parser.add_argument('-o', dest='csvpath', type=str, default='/Users/zongwei.zhou/Desktop/error_analysis',
                        help='the directory of the saved csv file recording all the error cases',
                       )
    parser.add_argument('--csvname', dest='csvname', type=str, default=None,
                        help='the directory of the saved csv file dataset statistics',
                       )
    parser.add_argument('--num_slices', action='store_true', default=False, 
                        help='check the number of slices in the z axis',
                       )
    parser.add_argument('--num_core', type=int, default=0, 
                        help='number of CPU core needed for this process'
                    )
    
    args = parser.parse_args()
    
    main(args)