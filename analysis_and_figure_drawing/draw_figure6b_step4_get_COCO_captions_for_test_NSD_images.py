from pycocotools.coco import COCO
import numpy as np

#cocoAPI
coco_ids_path = 'data/NSD_test_preprocessed/func1pt8mm/betas_fithrf_GLMdenoise_RR/subj01/val_cocoID_multi_trial_correct.npy'
pick_list = np.load('data/ROIs/corresponding_NSD_test_sample_index_Subject_1.npy')

annFile_trn = 'data/variables/captions_train2017.json'
annFile_val = 'data/variables/captions_val2017.json'

coco_caps_trn = COCO(annFile_trn)
coco_caps_val = COCO(annFile_val)
coco_ids = np.load(coco_ids_path)
coco_ids = coco_ids[pick_list]

with open('data/NSD_test_stimuli_shared_1k/captions_shared_1k.txt', 'w') as file:
    for i in range(len(pick_list)):
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
        print('i=', i+1)
        # print(caps)
        print(first_caption)