import numpy as np
import scipy.io as sio
import os
import h5py

roi_list = ['V1', 'V2', 'V3','V3ab','hV4','VO', 'PHC','LO','MT','MST','IPS','other']
for subjid in range(8):
    basepath = 'data/NSD_test_preprocessed/func1pt8mm/betas_fithrf_GLMdenoise_RR/subj0' + str(subjid + 1) + '/'
    savepath = 'data/ROIs/Subject_' + str(subjid + 1)
    os.makedirs(savepath, exist_ok=True)
    savepath = savepath + '/'
    pick_list = np.load('data/ROIs/corresponding_NSD_test_sample_index_Subject_' + str(subjid + 1) + '.npy')
    pick_list_1 = np.load('data/ROIs/corresponding_NSD_test_sample_index_Subject_' + str(subjid + 1) + '_trial_1.npy')
    pick_list_2 = np.load('data/ROIs/corresponding_NSD_test_sample_index_Subject_' + str(subjid + 1) + '_trial_2.npy')
    pick_list_3 = np.load('data/ROIs/corresponding_NSD_test_sample_index_Subject_' + str(subjid + 1) + '_trial_3.npy')
    roi_mask = np.load(basepath + 'roi_mask_V1.npy')
    roi_mask = np.transpose(roi_mask, (2, 1, 0)) # zyx-->xyz
    # build 4D array
    four_d_array = np.zeros((len(pick_list),) + roi_mask.shape)
    four_d_array_1 = np.zeros((len(pick_list_1),) + roi_mask.shape)
    four_d_array_2 = np.zeros((len(pick_list_2),) + roi_mask.shape)
    four_d_array_3 = np.zeros((len(pick_list_3),) + roi_mask.shape)
    print('subject=', subjid+1)
    for roi in roi_list:
        print('roi=', roi)
        fmri = np.load(basepath + 'val_voxel_single_trial_data_'+roi+'.npy')
        fmri_shared_1k_1 = fmri[pick_list_1, :]
        fmri_shared_1k_2 = fmri[pick_list_2, :]
        fmri_shared_1k_3 = fmri[pick_list_3, :]
        sio.savemat(savepath + 'fmri_shared_1k_'+roi+'_trial_1.mat', {'data': fmri_shared_1k_1})
        sio.savemat(savepath + 'fmri_shared_1k_' + roi + '_trial_2.mat', {'data': fmri_shared_1k_2})
        sio.savemat(savepath + 'fmri_shared_1k_' + roi + '_trial_3.mat', {'data': fmri_shared_1k_3})

        fmri = np.load(basepath + 'val_voxel_multi_trial_data_'+roi+'.npy')
        fmri_shared_1k = fmri[pick_list, :]
        sio.savemat(savepath + 'fmri_shared_1k_'+roi+'.mat', {'data': fmri_shared_1k})

        roi_mask = np.load(basepath + 'roi_mask_'+ roi +'.npy')
        roi_mask = np.transpose(roi_mask, (2, 1, 0)) # zyx-->xyz
        roi_ijk = np.argwhere(roi_mask != 0)
        print('voxel_num=',roi_ijk.shape[0])
        # np.save('data/NSD_test_preprocessed/func1pt8mm/betas_fithrf_GLMdenoise_RR/subj0' + str(subjid + 1) + '/'+ 'roi_ijk_'+roi+'.npy',roi_ijk)

        for voxel in range(fmri_shared_1k.shape[1]):
            i,j,k = roi_ijk[voxel,:]
            four_d_array[:,i,j,k]=fmri_shared_1k[:,voxel]
            four_d_array_1[:, i, j, k] = fmri_shared_1k_1[:, voxel]
            four_d_array_2[:, i, j, k] = fmri_shared_1k_2[:, voxel]
            four_d_array_3[:, i, j, k] = fmri_shared_1k_3[:, voxel]


    non_zero_count = np.count_nonzero(four_d_array)/fmri_shared_1k.shape[0]
    print('non_zero_count=',non_zero_count)

    # sio.savemat(savepath + 'fmri_shared_1k_all_rois_4d.mat', {'data': four_d_array}, do_compression=True) # in xyz format
    # sio.savemat(savepath + 'fmri_shared_1k_all_rois_4d_trial_1.mat', {'data': four_d_array_1}, do_compression=True)
    # sio.savemat(savepath + 'fmri_shared_1k_all_rois_4d_trial_2.mat', {'data': four_d_array_2}, do_compression=True)
    # sio.savemat(savepath + 'fmri_shared_1k_all_rois_4d_trial_3.mat', {'data': four_d_array_3}, do_compression=True)

    with h5py.File(savepath + 'fmri_shared_1k_all_rois_4d.h5', 'w') as f:
        f.create_dataset('data', data=four_d_array, compression="gzip")
    with h5py.File(savepath + 'fmri_shared_1k_all_rois_4d_trial_1.h5', 'w') as f:
        f.create_dataset('data', data=four_d_array_1, compression="gzip")
    with h5py.File(savepath + 'fmri_shared_1k_all_rois_4d_trial_2.h5', 'w') as f:
        f.create_dataset('data', data=four_d_array_2, compression="gzip")
    with h5py.File(savepath + 'fmri_shared_1k_all_rois_4d_trial_3.h5', 'w') as f:
        f.create_dataset('data', data=four_d_array_3, compression="gzip")
