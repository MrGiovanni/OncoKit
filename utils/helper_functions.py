import os
import cv2
import copy
import tarfile
import glob
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from skimage.measure import label
from tqdm import tqdm
from scipy import ndimage

# class map for the AbdomenAtlas 1.0 dataset
class_map_abdomenatlas_1_0 = {
    1: 'aorta',
    2: 'gall_bladder',
    3: 'kidney_left',
    4: 'kidney_right',
    5: 'liver',
    6: 'pancreas',
    7: 'postcava',
    8: 'spleen',
    9: 'stomach',
    }

# class map for the AbdomenAtlas 1.1 dataset
class_map_abdomenatlas_1_1 = {
    1: 'aorta', 
    2: 'gall_bladder', 
    3: 'kidney_left', 
    4: 'kidney_right', 
    5: 'liver', 
    6: 'pancreas', 
    7: 'postcava', 
    8: 'spleen', 
    9: 'stomach', 
    10: 'adrenal_gland_left', 
    11: 'adrenal_gland_right', 
    12: 'bladder', 
    13: 'celiac_trunk', 
    14: 'colon', 
    15: 'duodenum', 
    16: 'esophagus', 
    17: 'femur_left', 
    18: 'femur_right', 
    19: 'hepatic_vessel', 
    20: 'intestine', 
    21: 'lung_left', 
    22: 'lung_right', 
    23: 'portal_vein_and_splenic_vein', 
    24: 'prostate', 
    25: 'rectum'
    }

# class map for the AbdomenAtlas 1.2 dataset
class_map_abdomenatlas_1_2 = {
    1: 'aorta', 
    2: 'gall_bladder', 
    3: 'kidney_left', 
    4: 'kidney_right', 
    5: 'liver', 
    6: 'pancreas', 
    7: 'postcava', 
    8: 'spleen', 
    9: 'stomach', 
    10: 'adrenal_gland_left', 
    11: 'adrenal_gland_right', 
    12: 'bladder', 
    13: 'celiac_trunk', 
    14: 'colon', 
    15: 'duodenum', 
    16: 'esophagus', 
    17: 'femur_left', 
    18: 'femur_right', 
    19: 'hepatic_vessel', 
    20: 'intestine', 
    21: 'lung_left', 
    22: 'lung_right', 
    23: 'portal_vein_and_splenic_vein', 
    24: 'prostate', 
    25: 'rectum',
    26: 'vertebrae_L1', 
    27: 'vertebrae_L2', 
    28: 'vertebrae_L3', 
    29: 'vertebrae_L4', 
    30: 'vertebrae_L5', 
    31: 'vertebrae_T1', 
    32: 'vertebrae_T2', 
    33: 'vertebrae_T3', 
    34: 'vertebrae_T4', 
    35: 'vertebrae_T5', 
    36: 'vertebrae_T6', 
    37: 'vertebrae_T7', 
    38: 'vertebrae_T8', 
    39: 'vertebrae_T9', 
    40: 'vertebrae_T10', 
    41: 'vertebrae_T11', 
    42: 'vertebrae_T12', 
    43: 'vertebrae_C1', 
    44: 'vertebrae_C2', 
    45: 'vertebrae_C3', 
    46: 'vertebrae_C4', 
    47: 'vertebrae_C5', 
    48: 'vertebrae_C6', 
    49: 'vertebrae_C7',
    50: 'rib_left_1', 
    51: 'rib_left_2', 
    52: 'rib_left_3', 
    53: 'rib_left_4', 
    54: 'rib_left_5', 
    55: 'rib_left_6', 
    56: 'rib_left_7', 
    57: 'rib_left_8', 
    58: 'rib_left_9', 
    59: 'rib_left_10', 
    60: 'rib_left_11', 
    61: 'rib_left_12', 
    62: 'rib_right_1', 
    63: 'rib_right_2', 
    64: 'rib_right_3', 
    65: 'rib_right_4', 
    66: 'rib_right_5', 
    67: 'rib_right_6', 
    68: 'rib_right_7', 
    69: 'rib_right_8', 
    70: 'rib_right_9', 
    71: 'rib_right_10', 
    72: 'rib_right_11', 
    73: 'rib_right_12'
    }

# class map for the AbdomenAtlas 2.0 dataset
class_map_abdomenatlas_2_0 = {
    1: 'liver',
    2: 'liver_lesion',
    3: 'pancreas',
    4: 'pancreatic_lesion',
    5: 'kidney_left',
    6: 'kidney_right',
    7: 'kidney_lesion',
    8: 'kidney_tumor',
    9: 'kidney_cyst',
    10: 'colon',
    11: 'colon_lesion',
    12: 'uterus',
    13: 'endometrioma_tumor',
    14: 'esophagus',
    15: 'esophagus_tumor',
    }

# class map for the AbdomenAtlas X dataset
class_map_abdomenatlas_x = {
    1: 'lesion',
    }

# class map for the AbdomenAtlas ImageCAS dataset
class_map_abdomenatlas_imagecas = {
    1: 'coronary_artery',
    }

# class map for the AbdomenAtlas Report dataset
class_map_abdomenatlas_report = {
    1: 'pancreas',
    2: 'superior_mesenteric_artery',
    3: 'veins',
    4: 'celiac_aa',
    5: 'common_bile_duct',
    6: 'pancreatic_pdac',
    7: 'pancreatic_cyst',
    8: 'pancreatic_pnet',
    }

# class map for the AbdomenAtlas 3.0 dataset
class_map_abdomenatlas_3_0 = {
    1: 'liver',
    2: 'kidney_right',
    3: 'kidney_left',
    4: 'spleen',
    5: 'pancreas',
    6: 'pancreas_head',
    7: 'pancreas_body',
    8: 'pancreas_tail',
    9: 'liver_segment_1',
    10: 'liver_segment_2',
    11: 'liver_segment_3',
    12: 'liver_segment_4',
    13: 'liver_segment_5',
    14: 'liver_segment_6',
    15: 'liver_segment_7',
    16: 'liver_segment_8',
    17: 'colon',
    18: 'stomach',
    19: 'duodenum',
    20: 'common_bile_duct',
    21: 'intestine',
    22: 'aorta',
    23: 'postcava', # 'inferior vena cava'
    24: 'adrenal_gland_left',
    25: 'adrenal_gland_right',
    26: 'gall_bladder',
    27: 'bladder',
    28: 'celiac_trunk',
    29: 'esophagus',
    30: 'hepatic_vessel',
    31: 'portal_vein_and_splenic_vein',
    32: 'lung_left',
    33: 'lung_right',
    34: 'prostate',
    35: 'rectum',
    36: 'femur_left',
    37: 'femur_right',
    38: 'superior_mesenteric_artery',
    39: 'veins',
    40: 'liver_tumor',
    41: 'liver_cyst',
    42: 'liver_lesion',
    43: 'pancreatic_tumor',
    44: 'pancreatic_cyst',
    45: 'pancreatic_lesion',
    46: 'colon_tumor',
    47: 'colon_cyst',
    48: 'colon_lesion',
    49: 'kidney_tumor',
    50: 'kidney_cyst',
    51: 'kidney_lesion',
    52: 'pancreatic_pdac',
    53: 'pancreatic_pnet'
}

# class map for the AbdomenAtlas 3.1 dataset
class_map_abdomenatlas_3_1 = {
    1: 'liver',
    2: 'kidney_right',
    3: 'kidney_left',
    4: 'spleen',
    5: 'pancreas',
    6: 'pancreas_head',
    7: 'pancreas_body',
    8: 'pancreas_tail',
    9: 'liver_segment_1',
    10: 'liver_segment_2',
    11: 'liver_segment_3',
    12: 'liver_segment_4',
    13: 'liver_segment_5',
    14: 'liver_segment_6',
    15: 'liver_segment_7',
    16: 'liver_segment_8',
    17: 'colon',
    18: 'stomach',
    19: 'duodenum',
    20: 'common_bile_duct',
    21: 'intestine',
    22: 'aorta',
    23: 'postcava',
    24: 'adrenal_gland_left',
    25: 'adrenal_gland_right',
    26: 'gall_bladder',
    27: 'bladder',
    28: 'celiac_trunk',
    29: 'esophagus',
    30: 'hepatic_vessel',
    31: 'portal_vein_and_splenic_vein',
    32: 'lung_left',
    33: 'lung_right',
    34: 'lung_upper_left_lobe',
    35: 'lung_lower_left_lobe',
    36: 'lung_upper_right_lobe',
    37: 'lung_middle_right_lobe',
    38: 'lung_lower_right_lobe',
    39: 'prostate',
    40: 'rectum',
    41: 'femur_left',
    42: 'femur_right',
    43: 'superior_mesenteric_artery',
    44: 'veins',
    45: 'vertebrae_L1',
    46: 'vertebrae_L2',
    47: 'vertebrae_L3',
    48: 'vertebrae_L4',
    49: 'vertebrae_L5',
    50: 'vertebrae_T1',
    51: 'vertebrae_T2',
    52: 'vertebrae_T3',
    53: 'vertebrae_T4',
    54: 'vertebrae_T5',
    55: 'vertebrae_T6',
    56: 'vertebrae_T7',
    57: 'vertebrae_T8',
    58: 'vertebrae_T9',
    59: 'vertebrae_T10',
    60: 'vertebrae_T11',
    61: 'vertebrae_T12',
    62: 'vertebrae_C1',
    63: 'vertebrae_C2',
    64: 'vertebrae_C3',
    65: 'vertebrae_C4',
    66: 'vertebrae_C5',
    67: 'vertebrae_C6',
    68: 'vertebrae_C7',
    69: 'vertebrae_S1',
    70: 'rib_left_1',
    71: 'rib_left_2',
    72: 'rib_left_3',
    73: 'rib_left_4',
    74: 'rib_left_5',
    75: 'rib_left_6',
    76: 'rib_left_7',
    77: 'rib_left_8',
    78: 'rib_left_9',
    79: 'rib_left_10',
    80: 'rib_left_11',
    81: 'rib_left_12',
    82: 'rib_right_1',
    83: 'rib_right_2',
    84: 'rib_right_3',
    85: 'rib_right_4',
    86: 'rib_right_5',
    87: 'rib_right_6',
    88: 'rib_right_7',
    89: 'rib_right_8',
    90: 'rib_right_9',
    91: 'rib_right_10',
    92: 'rib_right_11',
    93: 'rib_right_12',
    94: 'trachea',
    95: 'iliac_artery_left',
    96: 'iliac_artery_right',
    97: 'iliac_vena_left',
    98: 'iliac_vena_right',
    99: 'humerus_left',
    100: 'humerus_right',
    101: 'scapula_left',
    102: 'scapula_right',
    103: 'clavicula_left',
    104: 'clavicula_right',
    105: 'hip_left',
    106: 'hip_right',
    107: 'sacrum',
    108: 'gluteus_maximus_left',
    109: 'gluteus_maximus_right',
    110: 'gluteus_medius_left',
    111: 'gluteus_medius_right',
    112: 'gluteus_minimus_left',
    113: 'gluteus_minimus_right',
    114: 'autochthon_left',
    115: 'autochthon_right',
    116: 'iliopsoas_left',
    117: 'iliopsoas_right',
    118: 'brachiocephalic_trunk',
    119: 'brachiocephalic_vein_left',
    120: 'brachiocephalic_vein_right',
    121: 'common_carotid_artery_left',
    122: 'common_carotid_artery_right',
    123: 'costal_cartilages',
    124: 'pulmonary_vein',
    125: 'subclavian_artery_left',
    126: 'subclavian_artery_right',
    127: 'superior_vena_cava',
    128: 'thyroid_gland',
    129: 'airway',
    130: 'skull',
    131: 'heart',
    132: 'brain',
    133: 'spinal_cord',
    134: 'sternum',
    135: 'atrial_appendage_left',
    136: 'liver_tumor',
    137: 'liver_cyst',
    138: 'liver_lesion',
    139: 'hepatic_tumor',
    140: 'pancreatic_tumor',
    141: 'pancreatic_cyst',
    142: 'pancreatic_lesion',
    143: 'pancreatic_pdac',
    144: 'pancreatic_pnet',
    145: 'colon_tumor',
    146: 'colon_cyst',
    147: 'colon_lesion',
    148: 'colon_cancer_primaries',
    149: 'kidney_tumor',
    150: 'kidney_cyst',
    151: 'kidney_lesion',
    152: 'lung_tumor',
    153: 'bone_lesion',
}

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def rename_delete_tumor_mask(pid, args):

    class_list = glob.glob(os.path.join(args.source_datapath, pid, 'segmentations', '*.nii.gz'))
    class_list = [c.split('/')[-1][:-7] for c in class_list]
    tumor_list = [c for c in class_list if ('tumor' in c or 'lesion' in c or 'cyst' in c or 'pdac' in c or 'pnet' in c) and c[0] != '_']
    psuedo_list = [c for c in class_list if c[0] == '_']

    # if kidney_tumor and kidney_lesion both exist in the tumor_list, delete kidney_lesion.nii.gz from destination folder
    if 'kidney_tumor' in tumor_list and 'kidney_lesion' in tumor_list:
        os.system('rm {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'kidney_lesion.nii.gz')))
        # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
        print('>> delete {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'kidney_lesion.nii.gz')))
    
    # if pid is greater than BDMAP_00005195 and kidney_tumor exists in the tumor_list, rename as kidney_lesion.nii.gz in destination folder
    if 'kidney_tumor' in tumor_list and int(pid.split('_')[-1]) > 5195 and '_A' not in pid and '_V' not in pid and '_O' not in pid:
        os.system('mv {} {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'kidney_tumor.nii.gz'), 
                                    os.path.join(args.destination_datapath, pid, 'segmentations', 'kidney_lesion.nii.gz')))
        # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
        print('>> rename {} as {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'kidney_tumor.nii.gz'), 
                                          os.path.join(args.destination_datapath, pid, 'segmentations', 'kidney_lesion.nii.gz')))
    
    # if liver_tumor and liver_lesion both exist in the tumor_list, delete liver_tumor.nii.gz from destination folder
    if 'liver_tumor' in tumor_list and 'liver_lesion' in tumor_list:
        os.system('rm {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'liver_tumor.nii.gz')))
        # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
        print('>> delete {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'liver_tumor.nii.gz')))

    # if liver_tumor exists in the tumor_list, rename as liver_lesion.nii.gz in destination folder
    if 'liver_tumor' in tumor_list:
        os.system('mv {} {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'liver_tumor.nii.gz'), 
                                    os.path.join(args.destination_datapath, pid, 'segmentations', 'liver_lesion.nii.gz')))
        # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
        print('>> rename {} as {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'liver_tumor.nii.gz'), 
                                          os.path.join(args.destination_datapath, pid, 'segmentations', 'liver_lesion.nii.gz')))

    if 'pancreas_tumor' in tumor_list:
        # remove pancreas_tumor.nii.gz from destination folder
        os.system('rm {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreas_tumor.nii.gz')))
        print('>> delete {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreas_tumor.nii.gz')))
        
    # if pancreatic_tumor and pancreatic_lesion both exist in the tumor_list, delete pancreatic_tumor.nii.gz from destination folder
    if 'pancreatic_tumor' in tumor_list and 'pancreatic_lesion' in tumor_list:
        os.system('rm {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreatic_tumor.nii.gz')))
        # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
        print('>> delete {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreatic_tumor.nii.gz')))

    # if pancreatic_tumor exists in the tumor_list, rename as pancreatic_lesion.nii.gz in destination folder
    if 'pancreatic_tumor' in tumor_list and 'pancreatic_pdac' not in tumor_list:
        os.system('mv {} {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreatic_tumor.nii.gz'), 
                                    os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreatic_lesion.nii.gz')))
        # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
        print('>> rename {} as {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreatic_tumor.nii.gz'), 
                                          os.path.join(args.destination_datapath, pid, 'segmentations', 'pancreatic_lesion.nii.gz')))

    # replace _xxx_tumor.nii.gz with _xxx_lesion.nii.gz in the destination folder
    for class_name in psuedo_list:
        if 'tumor' in class_name:
            os.system('mv {} {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', class_name + '.nii.gz'), 
                                        os.path.join(args.destination_datapath, pid, 'segmentations', class_name.replace('tumor', 'lesion') + '.nii.gz')))
            # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
            print('>> rename {} as {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', class_name + '.nii.gz'), 
                                                os.path.join(args.destination_datapath, pid, 'segmentations', class_name.replace('tumor', 'lesion') + '.nii.gz')))
    
    # if _xxx_tumor and xxx_tumor or xxx_lesion both exist in the tumor_list, delete _xxx_tumor.nii.gz from destination folder
    class_list = glob.glob(os.path.join(args.source_datapath, pid, 'segmentations', '*.nii.gz'))
    class_list = [c.split('/')[-1][:-7] for c in class_list]
    tumor_list = [c for c in class_list if ('tumor' in c or 'lesion' in c or 'cyst' in c or 'pdac' in c or 'pnet' in c) and c[0] != '_']
    psuedo_list = [c for c in class_list if c[0] == '_']
    for class_name in psuedo_list:
        if class_name[1:] in tumor_list:
            
            # delete the psuedo mask file in the destination folder
            os.system('rm {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', class_name + '.nii.gz')))
            # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
            print('>> delete {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', class_name + '.nii.gz')))
    
    # if _kidney_lesion exists in pseduo_list and kidney_tumor exists in tumor_list, delete _kidney_lesion.nii.gz from destination folder
    if 'kidney_tumor' in tumor_list and '_kidney_lesion' in psuedo_list:
        os.system('rm {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', '_kidney_lesion.nii.gz')))
        # print('\n>> processing {}\n{}\n{}'.format(pid, tumor_list, psuedo_list))
        print('>> delete {}'.format(os.path.join(args.destination_datapath, pid, 'segmentations', '_kidney_lesion.nii.gz')))

def count_num_slices(pid, datapath):

    # if there is a ct.nii.gz file
    if os.path.isfile(os.path.join(datapath, pid, 'segmentations', 'liver.nii.gz')):
        dim = get_dim(os.path.join(datapath, pid, 'segmentations', 'liver.nii.gz'))
    elif os.path.isfile(os.path.join(datapath, pid, 'ct.nii.gz')):
        dim = get_dim(os.path.join(datapath, pid, 'ct.nii.gz'))
    else:
        raise ValueError('No ct.nii.gz or liver.nii.gz file found in {}'.format(os.path.join(datapath, pid)))

    if (dim[0] == dim[1]) or \
    (dim[0] != dim[1] and dim[1] != dim[2]):
        return dim[-1]
    else:
        return dim[0]

def get_spacing(pid, datapath):

    # if there is a segmentations/liver.nii.gz file
    if os.path.isfile(os.path.join(datapath, pid, 'segmentations', 'liver.nii.gz')):
        nii = nib.load(os.path.join(datapath, pid, 'segmentations', 'liver.nii.gz'))
    elif os.path.isfile(os.path.join(datapath, pid, 'ct.nii.gz')):
        nii = nib.load(os.path.join(datapath, pid, 'ct.nii.gz'))
    else:
        raise ValueError('No ct.nii.gz or liver.nii.gz file found in {}'.format(os.path.join(datapath, pid)))
    
    spacing = nii.header['pixdim'][1:4]
    return spacing

def get_shape(pid, datapath):

    # if there is a ct.nii.gz file
    if os.path.isfile(os.path.join(datapath, pid, 'segmentations', 'liver.nii.gz')):
        dim = get_dim(os.path.join(datapath, pid, 'segmentations', 'liver.nii.gz'))
    elif os.path.isfile(os.path.join(datapath, pid, 'ct.nii.gz')):
        dim = get_dim(os.path.join(datapath, pid, 'ct.nii.gz'))
    else:
        raise ValueError('No ct.nii.gz or liver.nii.gz file found in {}'.format(os.path.join(datapath, pid)))
    
    return dim

def compute_centroid(mask):
    # Get the indices of the non-zero elements in the mask
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return None
    # Compute the centroid by averaging the indices
    centroid = np.mean(indices, axis=0)
    return centroid

def compute_distance(centroid1, centroid2):

    # Compute the Euclidean distance between two centroids
    distance = np.linalg.norm(centroid1 - centroid2)
    return distance

def generate_combined_labels(pid, datapath, class_maps):

    combined_labels, affine, header = load_mask(pid, 'liver', datapath)
    combined_labels.fill(0)
    for cid, class_name in class_maps.items():
        mask, _, _ = load_mask(pid, class_name, datapath)
        combined_labels[mask > 0.5] = cid
    nifti_path = os.path.join(datapath, pid, 'combined_labels.nii.gz')
    nib.save(nib.Nifti1Image(combined_labels.astype(np.uint8), affine=affine, header=header), nifti_path)

def getLargestCC(segmentation):
    labels = label(segmentation)
    if labels.max() > 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return largestCC
    else:
        return segmentation

def find_largest_subarray_bounds(arr, low_threshold, high_threshold):
    
    # Find the indices where the condition is True
    condition = (arr > low_threshold) & (arr < high_threshold)
    condition = getLargestCC(condition)
    x, y, z = np.where(condition)

    if not len(x):
        return (0,0,0), (0,0,0)  # No values above threshold

    # Find min and max indices along each dimension
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    min_z, max_z = np.min(z), np.max(z)

    return (min_x, min_y, min_z), (max_x, max_y, max_z)

def crop_largest_subarray(arr, low_threshold, high_threshold, case_name=None):
    
    (min_x, min_y, min_z), (max_x, max_y, max_z) = find_largest_subarray_bounds(arr, low_threshold, high_threshold)
    # if max_x - min_x < 50 or max_y - min_y < 50 or max_z - min_z < 5:
    #     print('ERROR in {}'.format(case_name))
    
    return (min_x, min_y, min_z), (max_x, max_y, max_z)

def standardization(original_ct_file, revised_ct_file, 
                    original_mask_file=None, revised_mask_file=None,
                    original_comb_file=None, revised_comb_file=None,
                    image_type=np.int16, mask_type=np.uint8,
                   ):
    
    img = nib.load(original_ct_file)
    data = np.array(img.dataobj)

    data[data > 1000] = 1000
    data[data < -1000] = -1000
    
    (min_x, min_y, min_z), (max_x, max_y, max_z) = crop_largest_subarray(arr=data, 
                                                                         low_threshold=-100, 
                                                                         high_threshold=100, 
                                                                         case_name=original_ct_file.split('/')[-2])
    data = data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

    data = nib.Nifti1Image(data, img.affine, img.header)
    data.set_data_dtype(image_type)
    data.get_data_dtype(finalize=True)
    
    nib.save(data, revised_ct_file)

    if original_comb_file is not None and revised_comb_file is not None:
        
        img = nib.load(original_comb_file)
        data = np.array(img.dataobj)
        mask = data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

        mask = nib.Nifti1Image(mask, img.affine, img.header)
        mask.set_data_dtype(mask_type)
        mask.get_data_dtype(finalize=True)

        nib.save(mask, revised_comb_file)

    if original_mask_file is not None and revised_mask_file is not None:
        
        for original, revised in zip(original_mask_file, revised_mask_file):
            img = nib.load(original)
            data = np.array(img.dataobj)
            mask = data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

            mask = nib.Nifti1Image(mask, img.affine, img.header)
            mask.set_data_dtype(mask_type)
            mask.get_data_dtype(finalize=True)

            nib.save(mask, revised)

def get_dim(nii_path):

    try:
        nii_image = nib.load(nii_path)
    except:
        print('{} is not a gzip file'.format(nii_path))
        sitk_img = sitk.ReadImage(nii_path)  # load with sitk
        sitk.WriteImage(sitk_img, nii_path)  # overwrite

    nii_image = nib.load(nii_path)
    
    return nii_image.shape

def load_ct(pid, datapath):

    ct_path = os.path.join(datapath, pid, 'ct.nii.gz')
    if os.path.isfile(ct_path):
        nii = nib.load(ct_path)
        ct = nii.get_fdata().astype(np.int16)
        return ct, nii.affine, nii.header
    else:
        None, None, None

def save_ct(data, affine, header, pid, datapath):
    
    if not os.path.exists(os.path.join(datapath, pid)):
        os.makedirs(os.path.join(datapath, pid))
    nifti_path = os.path.join(datapath, pid, 'ct.nii.gz')

    ct = nib.Nifti1Image(data, affine, header)
    ct.set_data_dtype(np.int16)
    ct.get_data_dtype(finalize=True)
    nib.save(ct, nifti_path)

def load_mask(pid, class_name, datapath, hiddenpath='/mnt/T9/AbdomenAtlasPro'):

    mask_path = os.path.join(datapath, pid, 'segmentations', class_name + '.nii.gz')
    mask_hidden_path = os.path.join(hiddenpath, pid, 'segmentations', class_name + '.nii.gz')
    if os.path.isfile(mask_path):
        nii = nib.load(mask_path)
        mask = nii.get_fdata().astype(np.uint8)
        return mask, nii.affine, nii.header
        
    elif os.path.isfile(mask_hidden_path):
        nii = nib.load(mask_hidden_path)
        mask = nii.get_fdata().astype(np.uint8)
        return mask, nii.affine, nii.header
        
    else:
        return None, None, None

def save_mask(data, affine, header, pid, class_name, datapath):
    
    if not os.path.exists(os.path.join(datapath, pid, 'segmentations')):
        os.makedirs(os.path.join(datapath, pid, 'segmentations'))
    nifti_path = os.path.join(datapath, pid, 'segmentations', class_name + '.nii.gz')

    mask = nib.Nifti1Image(data, affine, header)
    mask.set_data_dtype(np.uint8)
    mask.get_data_dtype(finalize=True)
    nib.save(mask, nifti_path)

def check_dim(list_of_array):

    dim = list_of_array[0].shape
    for i in range(len(list_of_array)):
        if dim != list_of_array[i].shape:
            return False
    return True

def plot_organ_projection(list_of_array, organ_name, pid, axis=2, pngpath=None):

    if axis == 2:
        projection = np.zeros((list_of_array[0][:,:,0].shape), dtype='float')
    else:
        raise
    for i in range(len(list_of_array)):
        organ_projection = np.sum(list_of_array[i], axis=axis) * 1.0
        organ_projection /= np.max(organ_projection)
        projection += organ_projection
    projection /= np.max(projection)
    projection *= 255.0
    projection = np.rot90(projection)

    if not os.path.exists(pngpath):
        os.makedirs(pngpath)
    cv2.imwrite(os.path.join(pngpath, pid + '.png'), projection)

