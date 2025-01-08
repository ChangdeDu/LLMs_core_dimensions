import clip
import torch
import numpy as np
from PIL import Image
from scipy import io
import glob
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Extract text features from Things
with open("data/variables/image_descriptions_meta_without_number_space.txt", "r") as file:
    lines = file.readlines()

text_features = []
for line in lines:
    text = clip.tokenize([line]).to(device)
    with torch.no_grad():
        text_features.append(model.encode_text(text).cpu().numpy())

# Save text features to a numpy array
text_feature_array = np.array(text_features)
np.save('data/variables/things1854image_caption_feas_CLIP_ViT_L14.npy', np.squeeze(text_feature_array))

# Extract image features from Things
image_features = []
for i in range(1854):
    image_path = f"data/THINGS_visual_stimuli_1854/image_{i+1}_ori.jpg"
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features.append(model.encode_image(image).cpu().numpy())

# Save image features as a numpy array
image_feature_array = np.array(image_features)
np.save('data/variables/things1854image_feas_CLIP_ViT_L14.npy',np.squeeze(image_feature_array))


# Extract image features from NSD shared_1k
image_features = []
shared_1k = 982
for i in range(shared_1k):
    image_path = f"data/NSD_test_stimuli_shared_1k/image_{i}.png"
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features.append(model.encode_image(image).cpu().numpy())
        
image_feature_array = np.array(image_features)
np.save('data/variables/shared_1k_image_feas_CLIP_ViT_L14.npy', np.squeeze(image_feature_array))
io.savemat('data/variables/shared_1k_image_feas_CLIP_ViT_L14.mat', {'data': image_feature_array})

# Extract text features from NSD shared_1k
with open("data/NSD_test_stimuli_shared_1k/captions_shared_1k.txt", "r") as file:
    lines = file.readlines()

text_features = []
for line in lines:
    text = clip.tokenize([line]).to(device)
    with torch.no_grad():
        text_features.append(model.encode_text(text).cpu().numpy())

text_feature_array = np.array(text_features)
np.save('data/variables/shared_1k_image_caption_feas_CLIP_ViT_L14.npy', np.squeeze(text_feature_array))
io.savemat('data/variables/shared_1k_image_caption_feas_CLIP_ViT_L14.mat', {'data': text_feature_array})


for subjid in [1, 2, 5, 7]:

    # Extract image features from NSD trn_stimuli
    folder_path = 'data/NSD_trn_stimuli/subj0' + str(subjid)
    png_pattern = os.path.join(folder_path, '*.png')
    png_files = glob.glob(png_pattern)
    num_png_files = len(png_files)
    image_features = []
    for i in range(num_png_files):
        image_path = f"data/NSD_trn_stimuli/subj0{str(subjid)}/image_{i}.png"
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features.append(model.encode_image(image).cpu().numpy())

    image_feature_array = np.array(image_features)
    np.save('data/variables/NSD_trn_image_feas_CLIP_ViT_L14'+ '_subj0' + str(subjid) + '.npy', np.squeeze(image_feature_array))
    # io.savemat('data/variables/NSD_trn_image_feas_CLIP_ViT_L14'+ '_subj0' + str(subjid) + '.mat', {'data': image_feature_array})

    # Extract text features from NSD trn_stimuli
    with open(folder_path+"/captions_trn.txt", "r") as file:
        lines = file.readlines()

    text_features = []
    for line in lines:
        text = clip.tokenize([line]).to(device)
        with torch.no_grad():
            text_features.append(model.encode_text(text).cpu().numpy())

    text_feature_array = np.array(text_features)
    np.save('data/variables/NSD_trn_image_caption_feas_CLIP_ViT_L14'+ '_subj0' + str(subjid) + '.npy', np.squeeze(text_feature_array))
    # io.savemat('data/variables/NSD_trn_image_caption_feas_CLIP_ViT_L14'+ '_subj0' + str(subjid) + '.mat', {'data': text_feature_array})