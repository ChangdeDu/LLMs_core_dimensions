import numpy as np
from PIL import Image

# human selected pick_list for subjects 1,2,5,7
# pick_list = [5,8,12,19,26,32,34,36,37,42,44,61,77,78,80,84,91,102,110,136,159,255,313,316,321,329,340,362,385,405,451,554,656,660,689,703,750,785,803,847,878,964,972,976]
pick_list = range(982)
for subjid in [1,2,5,7]:
    basepath = 'data/NSD_test_preprocessed/func1pt8mm/betas_fithrf_GLMdenoise_RR/subj0' + str(subjid) + '/'
    np.save('data/ROIs/corresponding_NSD_test_sample_index_Subject_' + str(subjid), pick_list)

images = np.load(basepath + 'val_stim_multi_trial_data.npy')
images_shared_1k = images[pick_list]
for i in range(images_shared_1k.shape[0]):
    img_array = np.transpose(images_shared_1k[i], (1, 2, 0)) * 255  # （256，256，3）
    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(f"data/NSD_test_stimuli_shared_1k/image_{i}.png")

# flatten images_shared_1k
images_shared_1k = images_shared_1k.reshape(images_shared_1k.shape[0], -1)
images_shared_1k = images_shared_1k[:,0:1000]

for subjid in [3,4,6,8]:
    basepath = 'data/NSD_test_preprocessed/func1pt8mm/betas_fithrf_GLMdenoise_RR/subj0' + str(subjid) + '/'
    images_multi_trial = np.load(basepath + 'val_stim_multi_trial_data.npy')
    images_multi_trial = images_multi_trial.reshape(images_multi_trial.shape[0], -1)
    images_multi_trial = images_multi_trial[:,0:1000]


    pick_list = []
    for index, row in enumerate(images_multi_trial):
        if any(np.array_equal(row, target_row) for target_row in images_shared_1k):
            pick_list.append(index)
    np.save('data/ROIs/corresponding_NSD_test_sample_index_Subject_' + str(subjid), pick_list)
