import cortex
import numpy as np
import scipy.io as sio
import h5py

np.random.seed(1234)
xfm = 'func1pt8_to_anat0pt8_autoFSbbr'

roi_list = ['EarlyVis','RSC', 'OPA', 'EBA', 'FFA-1','FFA-2', 'PPA','IPS','AG','TPOJ1','Broca']
subject_list =[1,2,5,7]
for sub in subject_list:
    subject = 'subj0' + str(sub) + '_ChangdeDuNMI'
    # Get the map of which voxels are inside of our ROI
    roi_masks = cortex.utils.get_roi_masks(subject, xfm,
                                            roi_list=roi_list,
                                            gm_sampler='cortical-conservative',
                                            # Select only voxels mostly within cortex
                                            split_lr=False,  # No separate left/right ROIs
                                            threshold=None,  # Leave roi mask values as probabilites / fractions
                                            return_dict=True
                                            )

    h5_file_path = 'data/ROIs/Subject_' + str(sub) + '/fmri_shared_1k_all_rois_4d.h5'
    with h5py.File(h5_file_path, 'r') as file:
        all_rois_4d_data = file['data'][:]
        print(all_rois_4d_data.shape)
        for roi in roi_list:
            mask = roi_masks[roi]
            mask[mask != 0] = 1
            mask = np.transpose(mask, (2, 1, 0))
            roi_array = np.array([sample[mask == 1] for sample in all_rois_4d_data])
            print(roi_array.shape)
            savepath = 'data/ROIs/Subject_' + str(sub)
            sio.savemat(savepath + '/fmri_shared_1k_'+roi +'.mat', {'data': roi_array})


