from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image

for subjid in [1,2,5,7]:
    folder_path = 'data/NSD_trn_stimuli/subj0' + str(subjid)
    os.makedirs(folder_path, exist_ok=True)

    basepath = 'data/NSD_test_preprocessed/func1pt8mm/betas_fithrf_GLMdenoise_RR/subj0' + str(subjid) + '/'
    images = np.load(basepath + 'trn_stim_data.npy')
    for i in range(images.shape[0]):
        img_array = np.transpose(images[i], (1, 2, 0)) * 255  # （256，256，3）
        img = Image.fromarray(img_array.astype(np.uint8))
        img.save(f"data/NSD_trn_stimuli/subj0{str(subjid)}/image_{i}.png")

    # cocoAPI
    coco_ids_path = 'data/NSD_test_preprocessed/func1pt8mm/betas_fithrf_GLMdenoise_RR/subj0' + str(subjid) + '/trn_cocoID_correct.npy'
    annFile_trn = 'data/variables/captions_train2017.json'
    annFile_val = 'data/variables/captions_val2017.json'
    coco_caps_trn = COCO(annFile_trn)
    coco_caps_val = COCO(annFile_val)
    coco_ids = np.load(coco_ids_path)

    with open('data/NSD_trn_stimuli/subj0' + str(subjid)+'/captions_trn.txt', 'w') as file:
        for i in range(images.shape[0]):
            coco_id = coco_ids[i]
            annIds = coco_caps_trn.getAnnIds(imgIds=coco_id)
            if len(annIds) == 0:
                annIds = coco_caps_val.getAnnIds(imgIds=coco_id)
                anns = coco_caps_val.loadAnns(annIds)
            else:
                anns = coco_caps_trn.loadAnns(annIds)

            caps = []  # Each image corresponds to five descriptions
            for j in range(len(anns)):
                cap = anns[j]['caption']
                caps.append(cap)

            # save the first description to a file
            if caps:  # Make sure the caps list is not empty
                first_caption = caps[0]
                first_caption = first_caption.splitlines()[0]
                file.write(f'{first_caption}\n')
            print('i=', i + 1)
            # print(caps)
            print(first_caption)