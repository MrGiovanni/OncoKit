'''
python -W ignore ct_quality_assessment.py --datapath /mnt/realccvl15/zzhou82/data/AbdomenAtlasPro/ --csvpath /mnt/T8/error_analysis/ --csvname ct_quality.csv
'''

import os
import argparse
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from helper_functions import *

def event(pid, args):

    data, affine, header = load_ct(pid, args.datapath)
    assert data is not None

    if np.min(data) < -1000 or np.max(data) > 1000:
        data[data<-1000] = -1000
        data[data>1000] = 1000
        print('>> min/max data error in {}, fixed'.format(pid))
    
        save_ct(data, affine, header,
                pid, args.datapath,
                )

        with open(os.path.join(args.csvpath, args.csvname), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([pid])

def main(args):

    if os.path.isfile(os.path.join(args.csvpath, args.csvname)):
        os.remove(os.path.join(args.csvpath, args.csvname))
    if not os.path.exists(args.csvpath):
        os.makedirs(args.csvpath)
    with open(os.path.join(args.csvpath, args.csvname), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Patient ID'])

    folder_names = [name for name in os.listdir(args.datapath) if os.path.isdir(os.path.join(args.datapath, name))]
    folder_names = sorted(folder_names)

    # folder_names = ['BDMAP_V0001120','BDMAP_V0000915']

    # for pid in tqdm(folder_names, ncols=80):
    #     event(pid, args)

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CT quality assessment')
    parser.add_argument('--datapath', type=str, required=True, help='Path to the data')
    parser.add_argument('--csvpath', type=str, required=True, help='Path to the csv file')
    parser.add_argument('--csvname', type=str, required=True, help='Name of the csv file')
    parser.add_argument('--num_core', type=int, default=0, help='number of CPU core needed for this process')

    args = parser.parse_args()

    main(args)