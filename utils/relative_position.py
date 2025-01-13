'''
source /data/zzhou82/environments/syn/bin/activate
python -W ignore relative_position.py --datapath /Volumes/T9/AbdomenAtlasProDemo --kidney
'''

import os
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from helper_functions import *

def event(pid, args):

    liver, _, _ = load_mask(pid, 'liver', args.datapath)
    spleen, _, _ = load_mask(pid, 'spleen', args.datapath)
    aorta, _, _ = load_mask(pid, 'aorta', args.datapath)
    postcava, _, _ = load_mask(pid, 'postcava', args.datapath)
    kidney_left, _, _ = load_mask(pid, 'kidney_left', args.datapath)
    kidney_right, _, _ = load_mask(pid, 'kidney_right', args.datapath)

    plot_organ_projection([liver, spleen], 
                           organ_name=['liver', 'spleen'],
                           pid=pid,
                           axis=2,
                           pngpath=args.pngpath,
                           )

def main(args):

    folder_names = [name for name in os.listdir(args.datapath) if os.path.isdir(os.path.join(args.datapath, name))]
    folder_names = sorted(folder_names)

    print('>> {} CPU cores are secured.'.format(cpu_count()))
    
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = {executor.submit(event, pid, args): pid
                   for pid in folder_names}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            folder = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {folder}: {e}")        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', dest='datapath', type=str, default='/Volumes/T9/AbdomenAtlasPro',
                        help='the directory of the AbdomenAtlas dataset',
                       )
    parser.add_argument('--pngpath', dest='pngpath', type=str, default='/Users/zongwei.zhou/Desktop/visual',
                        help='the directory to save png files',
                       )
    parser.add_argument('--aorta', action='store_true', default=False, 
                        help='check label quality for aorta?',
                       )
    parser.add_argument('--kidney', action='store_true', default=False, 
                        help='check label quality for kidney?',
                       )
    args = parser.parse_args()
    
    main(args)