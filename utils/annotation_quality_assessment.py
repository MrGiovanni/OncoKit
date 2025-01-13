'''
source /data/zzhou82/environments/syn/bin/activate
python -W ignore annotation_quality_assessment.py --datapath /Volumes/T9/AbdomenAtlasPro -o /Users/zongweizhou/Desktop/error_analysis --aorta --csvname aorta.csv
python -W ignore annotation_quality_assessment.py --datapath /Volumes/T9/AbdomenAtlasPro -o /Users/zongweizhou/Desktop/error_analysis --kidney --csvname kidney.csv
python -W ignore annotation_quality_assessment.py --datapath /Users/zongwei.zhou/Desktop/AbdomenAtlas1.1Mini -o /Users/zongwei.zhou/Desktop/error_analysis --prostate --csvname prostate.csv

python -W ignore annotation_quality_assessment.py --datapath /mnt/T9/AbdomenAtlasPro/ -o /mnt/T9/error_analysis/Pro --prostate --csvname prostate.csv >> /mnt/T9/error_analysis/Pro/prostate_log.txt
'''

import os
import argparse
import csv
from tqdm import tqdm
from medical_priors import *

def error_detection_per_case(pid, args):

    if args.adrenal_gland:
        return adrenal_gland_error(pid, args.datapath)
    if args.aorta:
        return aorta_error(pid, args.datapath)
    if args.femur:
        return femur_error(pid, args.datapath)
    if args.kidney:
        return kidney_error(pid, args.datapath)
    if args.liver:
        return liver_error(pid, args.datapath)
    if args.lung:
        return lung_error(pid, args.datapath)
    if args.prostate:
        return prostate_error(pid, args.datapath)
    if args.stomach:
        return stomach_error(pid, args.datapath)
    if args.spleen:
        return spleen_error(pid, args.datapath)

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
    error_list, full_list = [], []

    for pid in tqdm(folder_names[:9901]):
        full_list.append(pid)
        error_detected = error_detection_per_case(pid, args)
        if error_detected:
            error_list.append(pid)
            
            with open(os.path.join(args.csvpath, args.csvname), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([pid])

    print('\n> Overall error report {:.1f}% = {}/{}'.format(100.0*len(error_list)/len(full_list), len(error_list), len(full_list)))
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', dest='datapath', type=str, default='/Volumes/T9/AbdomenAtlasPro',
                        help='the directory of the AbdomenAtlas dataset',
                       )
    parser.add_argument('-o', dest='csvpath', type=str, default='/Users/zongwei.zhou/Desktop/error_analysis',
                        help='the directory of the saved csv file recording all the error cases',
                       )
    parser.add_argument('--csvname', dest='csvname', type=str, default='aorta.csv',
                        help='the directory of the saved csv file recording all the error cases',
                       )
    parser.add_argument('--adrenal_gland', action='store_true', default=False, 
                        help='check label quality for adrenal gland?',
                       )
    parser.add_argument('--aorta', action='store_true', default=False, 
                        help='check label quality for aorta?',
                       )
    parser.add_argument('--femur', action='store_true', default=False, 
                        help='check label quality for femur?',
                       )
    parser.add_argument('--kidney', action='store_true', default=False, 
                        help='check label quality for kidney?',
                       )
    parser.add_argument('--liver', action='store_true', default=False, 
                        help='check label quality for liver?',
                       )
    parser.add_argument('--lung', action='store_true', default=False, 
                        help='check label quality for lung?',
                       )
    parser.add_argument('--prostate', action='store_true', default=False, 
                        help='check label quality for prostate?',
                       )
    parser.add_argument('--stomach', action='store_true', default=False, 
                        help='check label quality for stomach?',
                       )
    parser.add_argument('--spleen', action='store_true', default=False, 
                        help='check label quality for spleen?',
                       )
    
    args = parser.parse_args()
    
    main(args)
