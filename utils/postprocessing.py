from helper_functions import *
from medical_priors import *

def kidney_postprocessing(pid, datapath):

    kidney_left, kidney_left_affine, kidney_left_header = load_mask(pid, 'kidney_left', datapath)
    kidney_right, kidney_right_affine, kidney_right_header = load_mask(pid, 'kidney_right', datapath)

    kidney_left_LargestCC = getLargestCC(kidney_left)
    kidney_right_LargestCC = getLargestCC(kidney_right)

    if np.sum(kidney_left - kidney_left_LargestCC) == 0:
        pass
    else:
        save_mask(kidney_left_LargestCC, kidney_left_affine, kidney_left_header, 
                  pid, 'kidney_left', datapath,
                  )
        print('kidney_left optimized in {}'.format(pid))

    if np.sum(kidney_right - kidney_right_LargestCC) == 0:
        pass
    else:
        save_mask(kidney_right_LargestCC, kidney_right_affine, kidney_right_header, 
                  pid, 'kidney_right', datapath,
                  )
        print('kidney_right optimized in {}'.format(pid))

def hepatic_vessel_postprocessing(pid, datapath):
    
    hepatic_vessel, hepatic_vessel_affine, hepatic_vessel_header = load_mask(pid, 'hepatic_vessel', datapath)
    assert hepatic_vessel is not None

    liver, liver_affine, liver_header = load_mask(pid, 'liver', datapath)
    assert liver is not None

    # 3D array liver dilated by N voxels outside
    liver_dilated = ndimage.binary_dilation(liver, structure=np.ones((5,5,5)))

    # for each voxel, if hepatic vessel has overlap with liver, then keep the hepatic vessel as one, otherwise set to zero.
    hepatic_vessel_revised = copy.deepcopy(hepatic_vessel)
    hepatic_vessel_revised[np.logical_not(liver_dilated)] = 0

    if np.sum(hepatic_vessel - hepatic_vessel_revised) == 0:
        pass
    else:
        save_mask(hepatic_vessel_revised, hepatic_vessel_affine, hepatic_vessel_header, 
                  pid, 'hepatic_vessel', datapath,
                 )
        print('hepatic_vessel optimized in {}'.format(pid))

def liver_postprocessing(pid, datapath):

    liver, liver_affine, liver_header = load_mask(pid, 'liver', datapath)
    assert liver is not None
    
    
    liver_LargestCC = getLargestCC(liver)
    
    if np.sum(liver - liver_LargestCC) == 0:
        pass
    else:
        save_mask(liver_LargestCC, liver_affine, liver_header, 
                  pid, 'liver', datapath,
                  )
        print('liver optimized in {}'.format(pid))

def spleen_postprocessing(pid, datapath):

    spleen, spleen_affine, spleen_header = load_mask(pid, 'spleen', datapath)
    assert spleen is not None
    
    
    spleen_LargestCC = getLargestCC(spleen)
    
    if np.sum(spleen - spleen_LargestCC) == 0:
        pass
    else:
        save_mask(spleen_LargestCC, spleen_affine, spleen_header, 
                  pid, 'spleen', datapath,
                  )
        print('spleen optimized in {}'.format(pid))

def stomach_postprocessing(pid, datapath):

    stomach, stomach_affine, stomach_header = load_mask(pid, 'stomach', datapath)
    assert stomach is not None
    
    
    stomach_LargestCC = getLargestCC(stomach)
    
    if np.sum(stomach - stomach_LargestCC) == 0:
        pass
    else:
        save_mask(stomach_LargestCC, stomach_affine, stomach_header, 
                  pid, 'stomach', datapath,
                  )
        print('stomach optimized in {}'.format(pid))

def gall_bladder_postprocessing(pid, datapath):

    gall_bladder, gall_bladder_affine, gall_bladder_header = load_mask(pid, 'gall_bladder', datapath)
    assert gall_bladder is not None
    
    
    gall_bladder_LargestCC = getLargestCC(gall_bladder)
    
    if np.sum(gall_bladder - gall_bladder_LargestCC) == 0:
        pass
    else:
        save_mask(gall_bladder_LargestCC, gall_bladder_affine, gall_bladder_header, 
                  pid, 'gall_bladder', datapath,
                  )
        print('gall_bladder optimized in {}'.format(pid))

def prostate_postprocessing(pid, datapath):

    prostate, prostate_affine, prostate_header = load_mask(pid, 'prostate', datapath)
    assert prostate is not None

    if prostate_error(pid, datapath, print_error=False):
        prostate.fill(0)
        save_mask(prostate, prostate_affine, prostate_header, 
                  pid, 'prostate', datapath,
                  )
        print('prostate prediction removed in {}'.format(pid))

    # prostate_LargestCC = getLargestCC(prostate)
    
    # if np.sum(prostate - prostate_LargestCC) == 0:
    #     pass
    # else:
    #     save_mask(prostate_LargestCC, prostate_affine, prostate_header, 
    #               pid, 'prostate', datapath,
    #               )
    #     print('prostate optimized in {}'.format(pid))

def postcava_postprocessing(pid, datapath):

    postcava, postcava_affine, postcava_header = load_mask(pid, 'postcava', datapath)
    assert postcava is not None
    
    postcava_LargestCC = getLargestCC(postcava)
    
    if np.sum(postcava - postcava_LargestCC) == 0:
        pass
    else:
        save_mask(postcava_LargestCC, postcava_affine, postcava_header, 
                  pid, 'postcava', datapath,
                  )
        print('postcava optimized in {}'.format(pid))
