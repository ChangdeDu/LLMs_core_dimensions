import numpy as np

for subjid in range(8):
    basepath = 'data/NSD_test_preprocessed/func1pt8mm/betas_fithrf_GLMdenoise_RR/subj0' + str(subjid + 1) + '/'
    images = np.load(basepath + 'val_stim_multi_trial_data.npy')
    images_multi_trial = np.load(basepath + 'val_stim_single_trial_data.npy')
    pick_list = np.load('data/ROIs/corresponding_NSD_test_sample_index_Subject_' + str(subjid + 1) + '.npy')
    images_shared_1k = images[pick_list]

    # flatten images_shared_1k and images_multi_trial
    images_shared_1k = images_shared_1k.reshape(images_shared_1k.shape[0], -1)
    images_shared_1k = images_shared_1k[:, 0:1000]

    images_multi_trial = images_multi_trial.reshape(images_multi_trial.shape[0], -1)
    images_multi_trial = images_multi_trial[:, 0:1000]

    matching_indices = []
    for index, row in enumerate(images_multi_trial):
        if any(np.array_equal(row, target_row) for target_row in images_shared_1k):
            matching_indices.append(index)

    images_matched = images_multi_trial[matching_indices, :]
    images_matched_mean = np.mean(images_matched, axis=1)
    images_shared_1k_mean = np.mean(images_shared_1k, axis=1)
    index_list_1 = []
    for i in range(images_shared_1k.shape[0]):
        indices = np.where(images_matched_mean == images_shared_1k_mean[i])[0][0]
        index_list_1.append(indices)

    index_list_2 = []
    for i in range(images_shared_1k.shape[0]):
        indices = np.where(images_matched_mean == images_shared_1k_mean[i])[0]
        if len(indices) > 1:
            index_list_2.append(indices[1])
        else:
            index_list_2.append(indices[0])

    index_list_3 = []
    for i in range(images_shared_1k.shape[0]):
        indices = np.where(images_matched_mean == images_shared_1k_mean[i])[0][-1]
        index_list_3.append(indices)

    np.save('data/ROIs/corresponding_NSD_test_sample_index_Subject_' + str(subjid + 1) + '_trial_1', index_list_1)
    np.save('data/ROIs/corresponding_NSD_test_sample_index_Subject_' + str(subjid + 1) + '_trial_2', index_list_2)
    np.save('data/ROIs/corresponding_NSD_test_sample_index_Subject_' + str(subjid + 1) + '_trial_3', index_list_3)