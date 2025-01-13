'''
python -W ignore generate_revision_list.py --csvpath /Users/zongwei.zhou/Dropbox\ \(ASU\)/PublicResource/Xinze/error_analysis/Pro --adrenal_gland --aorta  --femur --kidney --liver --lung --prostate --stomach --spleen
'''

import os
import argparse
import csv
from helper_functions import *

def read_csv_to_list(filename):
    data = []
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            data.append(row[0])
    print(filename, len(data))
    return data

def main(args):

    error_case_id = []
    if args.adrenal_gland:
        error_case_id.extend(read_csv_to_list(os.path.join(args.csvpath, 'adrenal_gland.csv')))
    if args.aorta:
        error_case_id.extend(read_csv_to_list(os.path.join(args.csvpath, 'aorta.csv')))
    if args.femur:
        error_case_id.extend(read_csv_to_list(os.path.join(args.csvpath, 'femur.csv')))
    if args.kidney:
        error_case_id.extend(read_csv_to_list(os.path.join(args.csvpath, 'kidney.csv')))
    if args.liver:
        error_case_id.extend(read_csv_to_list(os.path.join(args.csvpath, 'liver.csv')))
    if args.lung:
        error_case_id.extend(read_csv_to_list(os.path.join(args.csvpath, 'lung.csv')))
    if args.prostate:
        error_case_id.extend(read_csv_to_list(os.path.join(args.csvpath, 'prostate.csv')))
    if args.spleen:
        error_case_id.extend(read_csv_to_list(os.path.join(args.csvpath, 'spleen.csv')))
    if args.stomach:
        error_case_id.extend(read_csv_to_list(os.path.join(args.csvpath, 'stomach.csv')))
    
    all_id = read_csv_to_list(os.path.join(args.csvpath, 'all.csv'))
    weakly_revision_id = read_csv_to_list(os.path.join(args.csvpath, 'weakly_revision.csv'))

    quick_revision_id = list(set(all_id) - set(error_case_id))
    quick_revision_id = list(set(quick_revision_id) - set(weakly_revision_id))
    quick_revision_id.sort(reverse=True)

    if not os.path.exists(args.csvpath):
        os.makedirs(args.csvpath)
    if os.path.isfile(os.path.join(args.csvpath, args.csvname)):
        os.remove(os.path.join(args.csvpath, args.csvname))
    with open(os.path.join(args.csvpath, args.csvname), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Patient ID'])
    for pid in quick_revision_id:

        with open(os.path.join(args.csvpath, args.csvname), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([pid])

    # print(error_case_id, len(error_case_id))
    # print(all_id, len(all_id))
    print(len(quick_revision_id))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvpath', dest='csvpath', type=str, default='/Volumes/T9/error_analysis/Pro',
                        help='the directory of the csv files',
                       )
    parser.add_argument('--csvname', dest='csvname', type=str, default='quick_revision.csv',
                        help='the directory of the saved csv file recording all the cases to revise',
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
    parser.add_argument('--spleen', action='store_true', default=False, 
                        help='check label quality for spleen?',
                       )
    parser.add_argument('--stomach', action='store_true', default=False, 
                        help='check label quality for stomach?',
                       )
    
    
    args = parser.parse_args()
    
    main(args)