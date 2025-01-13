'''
source /data/zzhou82/environments/syn/bin/activate
python -W ignore annotation_refinement.py --datapath /mnt/T8/AbdomenAtlasPre/ --aorta --hepatic_vessel --prostate --postcava --kidney --liver --spleen --stomach --gall_bladder --postcava --kidney --liver --spleen --stomach --gall_bladder --postcava --num_core 40 >> /mnt/T8/stat/20240914_postprocessing_T8_AbdomenAtlasPre.txt

python -W ignore annotation_refinement.py --datapath /mnt/ccvl15/zzhou82/data/AbdomenAtlas/image_mask/AbdomenAtlas1.1Mini --prostate -v 1.1 --num_core 72 >> /mnt/T9/error_analysis/Pro/prostate_refinement.txt
'''

import os
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from postprocessing import *

def event(pid, args):

    if args.hepatic_vessel:
        hepatic_vessel_postprocessing(pid, args.datapath)
    if args.kidney:
        kidney_postprocessing(pid, args.datapath)
    if args.liver:
        liver_postprocessing(pid, args.datapath)
    if args.spleen:
        spleen_postprocessing(pid, args.datapath)
    if args.stomach:
        stomach_postprocessing(pid, args.datapath)
    if args.gall_bladder:
        gall_bladder_postprocessing(pid, args.datapath)
    if args.prostate:
        prostate_postprocessing(pid, args.datapath)
    if args.postcava:
        postcava_postprocessing(pid, args.datapath)
    
    if args.class_maps is not None:
        generate_combined_labels(pid, args.datapath, args.class_maps)

def main(args):

    if args.version == '1.0':
        args.class_maps = class_map_abdomenatlas_1_0
    elif args.version == '1.1':
        args.class_maps = class_map_abdomenatlas_1_1
    else:
        args.class_maps = None

    folder_names = [name for name in os.listdir(args.datapath) if os.path.isdir(os.path.join(args.datapath, name))]
    folder_names = sorted(folder_names)
    
    # TODO start delete
    # folder_names = ['BDMAP_00000001', 'BDMAP_00000002', 'BDMAP_00000003', 'BDMAP_00000004']
    # for pid in tqdm(folder_names):
    #     event(pid, args)
    # TODO end delete
    # TODO start delete
    folder_names = folder_names[:9901]
    # TODO end delete

    if args.num_core > 0:
        num_core = args.num_core
    else:
        num_core = int(cpu_count())

    print('>> {} CPU cores are secured.'.format(num_core))

    with ProcessPoolExecutor(max_workers=num_core) as executor:
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
    parser.add_argument('--num_core', dest='num_core', type=int, default=0,
                        help='number of CPU core needed for this process',
                       )
    parser.add_argument('--aorta', action='store_true', default=False, 
                        help='check label quality for aorta?',
                       )
    parser.add_argument('--hepatic_vessel', action='store_true', default=False,
                        help='check label quality for hepatic_vessel?',
                       )
    parser.add_argument('--kidney', action='store_true', default=False, 
                        help='check label quality for kidney?',
                       )
    parser.add_argument('--liver', action='store_true', default=False, 
                        help='check label quality for liver?',
                       )
    parser.add_argument('--spleen', action='store_true', default=False, 
                        help='check label quality for spleen?',
                       )
    parser.add_argument('--stomach', action='store_true', default=False, 
                        help='check label quality for stomach?',
                       )
    parser.add_argument('--gall_bladder', action='store_true', default=False, 
                        help='check label quality for gall_bladder?',
                       )
    parser.add_argument('--prostate', action='store_true', default=False, 
                        help='check label quality for prostate?',
                       )
    parser.add_argument('--postcava', action='store_true', default=False, 
                        help='check label quality for postcava?',
                       )
    parser.add_argument('-v', dest='version', type=str, default=None,
                        help='the version of dataset',
                       )
    args = parser.parse_args()
    
    main(args)