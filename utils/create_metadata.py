'''
python -W ignore create_metadata.py --datapath /mnt/T8/AbdomenAtlasPre -o /mnt/T8/metadata.csv
'''

import os
import argparse
import csv
from tqdm import tqdm
from helper_functions import *

def get_largest_tumor_diameter(pid, tumor_name, datapath):
    '''
    Get the largest diameter of the tumor for the patient
    '''
    tumor_diameters = []
    # if tumor nii file exists, get the largest tumor diameter
    if not os.path.exists(os.path.join(datapath, pid, 'segmentations', tumor_name+'.nii.gz')):
        return 0
    else:
        tumor_mask, tumor_affine, tumor_header = load_mask(pid, tumor_name, datapath)
        if np.sum(tumor_mask) == 0:
            return 0
        else:
            # get the largest tumor component in the mask
            tumor_LargestCC = getLargestCC(tumor_mask)
            spacing = tumor_header['pixdim'][1:4]
            tumor_volume = np.sum(tumor_LargestCC) * np.prod(spacing)
            tumor_diameter = 2 * ((3 * tumor_volume) / (4 * np.pi)) ** (1/3)

            return tumor_diameter / 10.0 # convert to cm

def main(args):

    if os.path.isfile(args.csvpath):
        os.remove(args.csvpath)
    if not os.path.exists(os.path.dirname(args.csvpath)):
        os.makedirs(os.path.dirname(args.csvpath))
    with open(args.csvpath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['BDMAP ID', 
                         'spacing', 
                         'shape', 
                         'sex',
                         'age',
                         'scanner',
                         'contrast',
                         'liver volume (cm^3)',
                         'total liver lesion volume (cm^3)',
                         'number of liver lesion instances',
                         'largest liver lesion diameter (cm)',
                         'largest liver cyst diameter (cm)',
                         'largest liver tumor diameter (cm)',
                         'largest liver lesion location (1-8)',
                         'largest liver lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)',
                         'pancreas volume (cm^3)',
                         'total pancreatic lesion volume (cm^3)',
                         'number of pancreatic lesion instances',
                         'largest pancreatic lesion diameter (cm)',
                         'largest pancreatic cyst diameter (cm)',
                         'largest pancreatic tumor diameter (cm)',
                         'largest pancreatic lesion location (head, body, tail)',
                         'largest pancreatic lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)',
                         'pancreatic tumor staging (T1-T4)',
                         'left kidney volume (cm^3)',
                         'right kidney volume (cm^3)',
                         'kidney volume (cm^3)',
                         'total kidney lesion volume (cm^3)',
                         'number of kidney lesion instances',
                         'largest kidney lesion diameter (cm)',
                         'largest kidney cyst diameter (cm)',
                         'largest kidney tumor diameter (cm)',
                         'largest kidney lesion location (left, right)',
                         'largest kidney lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)',
                         'spleen volume (cm^3)',
                         'total colon lesion volume (cm^3)',
                         'number of colon lesion instances',
                         'largest colon lesion diameter (cm)',
                         'total esophagus lesion volume (cm^3)',
                         'number of esophagus lesion instances',
                         'largest esophagus lesion diameter (cm)',
                         'total uterus lesion volume (cm^3)',
                         'number of uterus lesion instances',
                         'largest uterus lesion diameter (cm)',
                         'structured report',
                         'narrative report',
                         ])
    
    folder_names = [name for name in os.listdir(args.datapath) if os.path.isdir(os.path.join(args.datapath, name))]
    folder_names = sorted(folder_names)

    # folder_names = folder_names[:10]
    
    for pid in tqdm(folder_names, ncols=80):

        shape = get_shape(pid, args.datapath)
        spacing = get_spacing(pid, args.datapath)
        largest_liver_lesion_diameter = max(get_largest_tumor_diameter(pid, 'liver_lesion', args.datapath),
                                           get_largest_tumor_diameter(pid, 'liver_tumor', args.datapath),
                                           )   
        largest_pancreatic_lesion_diameter = max(get_largest_tumor_diameter(pid, 'pancreatic_lesion', args.datapath),
                                                get_largest_tumor_diameter(pid, 'pancreatic_pdac', args.datapath),
                                                get_largest_tumor_diameter(pid, 'pancreatic_pnet', args.datapath),
                                                get_largest_tumor_diameter(pid, 'pancreatic_cyst', args.datapath),
                                                get_largest_tumor_diameter(pid, 'pancreatic_tumor', args.datapath),
                                                )
        largest_kidney_lesion_diameter = max(get_largest_tumor_diameter(pid, 'kidney_lesion', args.datapath),
                                            get_largest_tumor_diameter(pid, 'kidney_cyst', args.datapath),
                                            get_largest_tumor_diameter(pid, 'kidney_tumor', args.datapath),
                                            )
        largest_colon_lesion_diameter = max(get_largest_tumor_diameter(pid, 'colon_lesion', args.datapath),
                                            get_largest_tumor_diameter(pid, 'colon_tumor', args.datapath),
                                            )
        largest_esophagus_lesion_diameter = get_largest_tumor_diameter(pid, 'esophagus_lesion', args.datapath)
        largest_uterus_lesion_diameter = get_largest_tumor_diameter(pid, 'uterus_lesion', args.datapath)

        with open(args.csvpath, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # write to correponding columns based on the csv header - for example, 
            # largest_liver_lesion_diameter in 'largest liver lesion diameter (cm)'
            # largest_pancreatic_lesion_diameter in 'largest pancreatic lesion diameter (cm)'
            # largest_kidney_lesion_diameter in 'largest kidney lesion diameter (cm)'
            # largest_colon_lesion_diameter in 'largest colon lesion diameter (cm)'
            # largest_esophagus_lesion_diameter in 'largest esophagus lesion diameter (cm)'
            # largest_uterus_lesion_diameter in 'largest uterus lesion diameter (cm)'
            writer.writerow([pid, 
                             spacing, 
                             shape,
                             '',  # Placeholder for 'sex'
                             '',  # Placeholder for 'age'
                             '',  # Placeholder for 'scanner'
                             '',  # Placeholder for 'contrast'
                             '',  # Placeholder for 'liver volume (cm^3)'
                             '',  # Placeholder for 'total liver lesion volume (cm^3)'
                             '',  # Placeholder for 'number of liver lesion instances'
                             largest_liver_lesion_diameter,
                             '',  # Placeholder for 'largest liver cyst diameter (cm)'
                             '',  # Placeholder for 'largest liver tumor diameter (cm)'
                             '',  # Placeholder for 'largest liver lesion location (1-8)'
                             '',  # Placeholder for 'largest liver lesion attenuation'
                             '',  # Placeholder for 'pancreas volume (cm^3)'
                             '',  # Placeholder for 'total pancreatic lesion volume (cm^3)'
                             '',  # Placeholder for 'number of pancreatic lesion instances'
                             largest_pancreatic_lesion_diameter,
                             '',  # Placeholder for 'largest pancreatic cyst diameter (cm)'
                             '',  # Placeholder for 'largest pancreatic tumor diameter (cm)'
                             '',  # Placeholder for 'largest pancreatic lesion location (head, body, tail)'
                             '',  # Placeholder for 'largest pancreatic lesion attenuation'
                             '',  # Placeholder for 'pancreatic tumor staging (T1-T4)'
                             '',  # Placeholder for 'left kidney volume (cm^3)'
                             '',  # Placeholder for 'right kidney volume (cm^3)'
                             '',  # Placeholder for 'kidney volume (cm^3)'
                             '',  # Placeholder for 'total kidney lesion volume (cm^3)'
                             '',  # Placeholder for 'number of kidney lesion instances'
                             largest_kidney_lesion_diameter,
                             '',  # Placeholder for 'largest kidney cyst diameter (cm)'
                             '',  # Placeholder for 'largest kidney tumor diameter (cm)'
                             '',  # Placeholder for 'largest kidney lesion location (left, right)'
                             '',  # Placeholder for 'largest kidney lesion attenuation'
                             '',  # Placeholder for 'spleen volume (cm^3)'
                             '',  # Placeholder for 'total colon lesion volume (cm^3)'
                             '',  # Placeholder for 'number of colon lesion instances'
                             largest_colon_lesion_diameter,
                             '',  # Placeholder for 'total esophagus lesion volume (cm^3)'
                             '',  # Placeholder for 'number of esophagus lesion instances'
                             largest_esophagus_lesion_diameter,
                             '',  # Placeholder for 'total uterus lesion volume (cm^3)'
                             '',  # Placeholder for 'number of uterus lesion instances'
                             largest_uterus_lesion_diameter,
                             '',  # Placeholder for 'structured report'
                             ''   # Placeholder for 'narrative report'
                             ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', 
                        dest='datapath', 
                        type=str, 
                        default='/mnt/T8/AbdomenAtlasPre',
                        help='the directory of the AbdomenAtlas dataset',
                       )
    parser.add_argument('-o', 
                        dest='csvpath', 
                        type=str, 
                        default='/mnt/T8/metadata.csv',
                        help='the directory of the saved csv file recording all the error cases',
                       )
    args = parser.parse_args()
    main(args)