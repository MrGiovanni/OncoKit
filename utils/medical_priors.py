from helper_functions import *

def adrenal_gland_error(pid, datapath, minimal_overlap=0):

    '''
    Overlap rule:
        1. no overlap between left and right adrenal gland
        2. D(adrenal_gland_left, liver) > D(adrenal_gland_right, liver)
    '''

    error_detected = False

    adrenal_gland_left, _, _ = load_mask(pid, 'adrenal_gland_left', datapath)
    adrenal_gland_right, _, _ = load_mask(pid, 'adrenal_gland_right', datapath)
    if adrenal_gland_left is None or adrenal_gland_right is None:
        return error_detected
    
    liver, _, _ = load_mask(pid, 'liver', datapath)

    assert check_dim([adrenal_gland_left, adrenal_gland_right])

    adrenal_gland_lr_overlap = error3d_overlaps(adrenal_gland_left, adrenal_gland_right)
    if adrenal_gland_lr_overlap > minimal_overlap:
        print('> {} has {} px overlap betwee adrenal gland L&R'.format(pid, adrenal_gland_lr_overlap))
        error_detected = True

    if liver is None:
        return error_detected
    # Compute centroids
    centroid_left = compute_centroid(adrenal_gland_left)
    centroid_right = compute_centroid(adrenal_gland_right)
    centroid_liver = compute_centroid(liver)
    
    if centroid_left is not None and centroid_right is not None and centroid_liver is not None:
        distance_liver_left = compute_distance(centroid_liver, centroid_left)
        distance_liver_right = compute_distance(centroid_liver, centroid_right)
        if distance_liver_left < distance_liver_right:
            print('> {} has wrong L&R adrenal gland'.format(pid))
            error_detected = True

    return error_detected

def aorta_error(pid, datapath):

    '''
    Overlap rules:
        1. [removed] if liver exists, aorta must exist
        2. if postcava and lung exist, aorta must exist
    '''

    aorta, _, _ = load_mask(pid, 'aorta', datapath)
    lung_left, _, _ = load_mask(pid, 'lung_left', datapath)
    lung_right, _, _ = load_mask(pid, 'lung_right', datapath)
    postcava, _, _ = load_mask(pid, 'postcava', datapath)

    if aorta is None or lung_left is None or lung_right is None or postcava is None:
        return False

    assert check_dim([aorta, lung_left, lung_right, postcava])

    error_count = 0

    if (aorta.shape[0] == aorta.shape[1]) or \
    (aorta.shape[0] != aorta.shape[1] and aorta.shape[1] != aorta.shape[2]):
        for i in range(aorta.shape[-1]):
            if error2d_ispostcava_islung_nonaorta(postcava[:,:,i], lung_right[:,:,i], lung_left[:,:,i], aorta[:,:,i]):
                error_count += 1
        error_percent = 100.0*error_count/aorta.shape[-1]
        print('> {} has {:.1f}% ({}/{}) errors in aorta'.format(pid,
                                                                error_percent,
                                                                error_count,
                                                                aorta.shape[-1],
                                                                ))
    
    else:
        for i in range(aorta.shape[0]):
            if error2d_ispostcava_islung_nonaorta(postcava[i,:,:], lung_right[i,:,:], lung_left[i,:,:], aorta[i,:,:]):
                error_count += 1
        error_percent = 100.0*error_count/aorta.shape[0]
        print('> {} has {:.1f}% ({}/{}) errors in aorta'.format(pid,
                                                                error_percent,
                                                                error_count,
                                                                aorta.shape[0],
                                                                ))
    if error_percent > 3:
        error_detected = True
    else:
        error_detected = False

    return error_detected

def femur_error(pid, datapath, minimal_overlap=0):

    '''
    Overlap rule:
        1. no overlap between left and right femur
        2. D(femur_left, liver) > D(femur_right, liver)
    '''

    error_detected = False

    femur_left, _, _ = load_mask(pid, 'femur_left', datapath)
    femur_right, _, _ = load_mask(pid, 'femur_right', datapath)
    if femur_left is None or femur_right is None:
        return error_detected
    

    assert check_dim([femur_left, femur_right])

    femur_lr_overlap = error3d_overlaps(femur_left, femur_right)
    if femur_lr_overlap > minimal_overlap:
        print('> {} has {} px overlap betwee femur L&R'.format(pid, femur_lr_overlap))
        error_detected = True

    # Compute centroids
    liver, _, _ = load_mask(pid, 'liver', datapath)
    if liver is None:
        return error_detected
    
    centroid_left = compute_centroid(femur_left)
    centroid_right = compute_centroid(femur_right)
    centroid_liver = compute_centroid(liver)
    
    if centroid_left is not None and centroid_right is not None and centroid_liver is not None:
        distance_liver_left = compute_distance(centroid_liver, centroid_left)
        distance_liver_right = compute_distance(centroid_liver, centroid_right)
        if distance_liver_left < distance_liver_right:
            print('> {} has wrong L&R femur'.format(pid))
            error_detected = True

    return error_detected

def kidney_error(pid, datapath, minimal_overlap=0):

    '''
    Overlap rule:
        1. no overlap between left and right kidneys
        2. D(kidney_left, liver) > D(kidney_right, liver)
    '''

    error_detected = False

    kidney_left, _, _ = load_mask(pid, 'kidney_left', datapath)
    kidney_right, _, _ = load_mask(pid, 'kidney_right', datapath)
    if kidney_left is None or kidney_right is None:
        return error_detected
    liver, _, _ = load_mask(pid, 'liver', datapath)

    assert check_dim([kidney_left, kidney_right])

    kidney_lr_overlap = error3d_overlaps(kidney_left, kidney_right)
    if kidney_lr_overlap > minimal_overlap:
        print('> {} has {} px overlap betwee kidney L&R'.format(pid, kidney_lr_overlap))
        error_detected = True

    if liver is None:
        return error_detected
    # Compute centroids
    centroid_left = compute_centroid(kidney_left)
    centroid_right = compute_centroid(kidney_right)
    centroid_liver = compute_centroid(liver)
    
    if centroid_left is not None and centroid_right is not None and centroid_liver is not None:
        distance_liver_left = compute_distance(centroid_liver, centroid_left)
        distance_liver_right = compute_distance(centroid_liver, centroid_right)
        if distance_liver_left < distance_liver_right:
            print('> {} has wrong L&R kidneys'.format(pid))
            error_detected = True

    return error_detected

def liver_error(pid, datapath, minimal_overlap=10):

    '''
    Overlap rules:
        1. liver has no overlap with left kidney
        2. liver has no overlap with aorta
    '''

    error_detected = False

    liver, _, _ = load_mask(pid, 'liver', datapath)
    if liver is None:
        return error_detected
    kidney_left, _, _ = load_mask(pid, 'kidney_left', datapath)
    aorta, _, _ = load_mask(pid, 'aorta', datapath)

    if kidney_left is not None:
        assert check_dim([liver, kidney_left])
        overlap = error3d_overlaps(liver, kidney_left)
        if overlap > minimal_overlap:
            print('> {} has {} px overlap betwee liver & kidney L'.format(pid, overlap))
            error_detected = True
    
    if aorta is not None:
        assert check_dim([liver, aorta])
        overlap = error3d_overlaps(liver, aorta)
        if overlap > minimal_overlap:
            print('> {} has {} px overlap betwee liver & aorta'.format(pid, overlap))
            error_detected = True

    return error_detected

def lung_error(pid, datapath, minimal_overlap=10):

    '''
    Overlap rule:
        1. no overlap between left and right lung
        2. D(lung_left, liver) > D(lung_right, liver)
    '''

    error_detected = False

    lung_left, _, _ = load_mask(pid, 'lung_left', datapath)
    lung_right, _, _ = load_mask(pid, 'lung_right', datapath)
    if lung_left is None or lung_right is None:
        return error_detected
    assert check_dim([lung_left, lung_right])

    lung_lr_overlap = error3d_overlaps(lung_left, lung_right)
    if lung_lr_overlap > minimal_overlap:
        print('> {} has {} px overlap betwee lung L&R'.format(pid, lung_lr_overlap))
        error_detected = True

    # Compute centroids
    liver, _, _ = load_mask(pid, 'liver', datapath)
    if liver is None:
        return error_detected
    
    centroid_left = compute_centroid(lung_left)
    centroid_right = compute_centroid(lung_right)
    centroid_liver = compute_centroid(liver)
    
    if centroid_left is not None and centroid_right is not None and centroid_liver is not None:
        distance_liver_left = compute_distance(centroid_liver, centroid_left)
        distance_liver_right = compute_distance(centroid_liver, centroid_right)
        if distance_liver_left < distance_liver_right:
            print('> {} has wrong L&R lung'.format(pid))
            error_detected = True

    return error_detected

def prostate_error(pid, datapath, minimal_detect=4, print_error=True):

    '''
    Overlap rules:
        1. if kidney and femur exist, prostate must exist
        2. prostate locates in between of L&R kidney in X-axis
        3. prostate locates in between of L&R femur in X-axis
        4 [?]. prostate locates in between of kidney and femur in Z-axis
        4. Dz(prostate, kidney) > Dz(prostate, femur)
    '''

    error_detected = False

    prostate, _, _ = load_mask(pid, 'prostate', datapath)
    if prostate is None:
        return error_detected
    kidney_left, _, _ = load_mask(pid, 'kidney_left', datapath)
    kidney_right, _, _ = load_mask(pid, 'kidney_right', datapath)
    femur_left, _, _ = load_mask(pid, 'femur_left', datapath)
    femur_right, _, _ = load_mask(pid, 'femur_right', datapath)

    if kidney_left is not None and kidney_right is not None and femur_left is not None and femur_right is not None:
        assert check_dim([prostate, kidney_left, kidney_right, femur_left, femur_right])

        if np.sum(prostate) == 0 and (np.sum(kidney_left) > minimal_detect or np.sum(kidney_right) > minimal_detect) and (np.sum(femur_left) > minimal_detect or np.sum(femur_right) > minimal_detect):
            if print_error:
                print('> {} misses prostate annotation'.format(pid))
            error_detected = True
        
        if np.sum(prostate) > minimal_detect and np.sum(kidney_left) > minimal_detect and np.sum(kidney_right) > minimal_detect and np.sum(femur_left) > minimal_detect and np.sum(femur_right) > minimal_detect:
            centroid_kidney_left = compute_centroid(kidney_left)
            centroid_kidney_right = compute_centroid(kidney_right)
            centroid_femur_left = compute_centroid(femur_left)
            centroid_femur_right = compute_centroid(femur_right)
            centroid_prostate = compute_centroid(prostate)

            # determine which is X axis
            abs_centroid = np.zeros((3,), dtype=float)
            for i in range(len(prostate.shape)):
                abs_centroid[i] = abs(centroid_kidney_left[i] - centroid_kidney_right[i])
            X_axis = np.argmax(abs_centroid)
            Z_axis = 2 - X_axis
            
            if ((centroid_prostate[X_axis] - centroid_kidney_left[X_axis]) * (centroid_prostate[X_axis] - centroid_kidney_right[X_axis]) > 0) or \
               ((centroid_prostate[X_axis] - centroid_femur_left[X_axis]) * (centroid_prostate[X_axis] - centroid_femur_right[X_axis]) > 0):
                if print_error:
                    print('> {} has wrong prostate location in X-axis'.format(pid))
                error_detected = True
            
            kidney_z = (centroid_kidney_left[Z_axis] + centroid_kidney_right[Z_axis])/2
            femur_z = (centroid_femur_left[Z_axis] + centroid_femur_right[Z_axis])/2
            if abs(centroid_prostate[Z_axis] - kidney_z) < abs(centroid_prostate[Z_axis] - femur_z):
                if print_error:
                    print('> {} has wrong prostate location in Z-axis'.format(pid))
                error_detected = True

    return error_detected

def spleen_error(pid, datapath, minimal_overlap=10):

    '''
    Overlap rules:
        1. spleen has no overlap with right kidney
        2. spleen has no overlap with gall bladder
    '''

    error_detected = False

    spleen, _, _ = load_mask(pid, 'spleen', datapath)
    if spleen is None:
        return error_detected
    kidney_right, _, _ = load_mask(pid, 'kidney_right', datapath)
    gall_bladder, _, _ = load_mask(pid, 'gall_bladder', datapath)

    if kidney_right is not None:
        assert check_dim([spleen, kidney_right])
        overlap = error3d_overlaps(spleen, kidney_right)
        if overlap > minimal_overlap:
            print('> {} has {} px overlap betwee spleen & kidney R'.format(pid, overlap))
            error_detected = True
    
    if gall_bladder is not None:
        assert check_dim([spleen, gall_bladder])
        overlap = error3d_overlaps(spleen, gall_bladder)
        if overlap > minimal_overlap:
            print('> {} has {} px overlap betwee spleen & gall_bladder'.format(pid, overlap))
            error_detected = True

    return error_detected

def stomach_error(pid, datapath, minimal_overlap=10):

    '''
    Overlap rules:
        1. stomach has no overlap with left kidney
        2. stomach has no overlap with aorta
        3. stomach has no overlap with postcava
    '''

    error_detected = False

    stomach, _, _ = load_mask(pid, 'stomach', datapath)
    if stomach is None:
        return error_detected
    kidney_left, _, _ = load_mask(pid, 'kidney_left', datapath)
    aorta, _, _ = load_mask(pid, 'aorta', datapath)
    postcava, _, _ = load_mask(pid, 'postcava', datapath)

    if kidney_left is not None:
        assert check_dim([stomach, kidney_left])
        overlap = error3d_overlaps(stomach, kidney_left)
        if overlap > minimal_overlap:
            print('> {} has {} px overlap betwee stomach & kidney L'.format(pid, overlap))
            error_detected = True
    
    if aorta is not None:
        assert check_dim([stomach, aorta])
        overlap = error3d_overlaps(stomach, aorta)
        if overlap > minimal_overlap:
            print('> {} has {} px overlap betwee stomach & aorta'.format(pid, overlap))
            error_detected = True

    if postcava is not None:
        assert check_dim([stomach, postcava])
        overlap = error3d_overlaps(stomach, postcava)
        if overlap > minimal_overlap:
            print('> {} has {} px overlap betwee stomach & postcava'.format(pid, overlap))
            error_detected = True

    return error_detected

def error2d_isliver_noaorta(liver, aorta):

    if liver is None or aorta is None:
        return False

    if np.sum(liver) > 0 and np.sum(aorta) == 0:
        return True
    else:
        return False
    
def error2d_ispostcava_islung_nonaorta(postcava, lung_right, lung_left, aorta):

    if postcava is None or lung_right is None or lung_left is None or aorta is None:
        return False
    
    if np.sum(postcava) > 0 and (np.sum(lung_right) > 0 or np.sum(lung_left) > 0) and np.sum(aorta) == 0:
        return True
    else:
        return False
    
def error3d_overlaps(c1, c2):

    if c1 is None or c2 is None:
        return False
    
    return np.sum(np.logical_and(c1, c2))

