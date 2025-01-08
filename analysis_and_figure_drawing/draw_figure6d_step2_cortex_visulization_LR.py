import cortex
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
np.random.seed(1234)

sub = 1 #2,5,7
subject = 'subj0'+str(sub)+'_ChangdeDuNMI'
xfm = 'func1pt8_to_anat0pt8_autoFSbbr'
metric = 'R2'

#### for Human
mat_file_path = f'data/ROIs/Subject_{sub}/fmri_shared_1k_all_rois_3d_{metric}_humans.mat'
mat_contents = loadmat(mat_file_path)
test_data_Human=mat_contents['data']

mat_file_path = f'data/ROIs/Subject_{sub}/fmri_shared_1k_all_rois_3d_nc.mat'
mat_contents = loadmat(mat_file_path)
test_data_nc=mat_contents['data']
test_data_nc[test_data_nc < 20] = 100
test_data_Human = 100 * test_data_Human / test_data_nc

test_data = np.transpose(test_data_Human, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_Human)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)
cortex.quickshow(vol_data, height=1024, dpi=100, with_labels=True, with_colorbar=True, with_rois=True,labelsize="50pt", with_curvature=True, curvature_brightness=0.5,
    curvature_contrast=0.25)
plt.savefig(f'figures/Subject_{sub}_humans_{metric}.png')

#### for MLLM
mat_file_path = f'data/ROIs/Subject_{sub}/fmri_shared_1k_all_rois_3d_{metric}_gemini.mat'
mat_contents = loadmat(mat_file_path)
test_data_MLLM=mat_contents['data']

mat_file_path = f'data/ROIs/Subject_{sub}/fmri_shared_1k_all_rois_3d_nc.mat'
mat_contents = loadmat(mat_file_path)
test_data_nc=mat_contents['data']
test_data_nc[test_data_nc < 20] = 100
test_data_MLLM = 100 * test_data_MLLM / test_data_nc

test_data = np.transpose(test_data_MLLM, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_Human)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)
cortex.quickshow(vol_data, height=1024, dpi=100, with_labels=True, with_colorbar=True, with_rois=True,labelsize="50pt", with_curvature=True, curvature_brightness=0.5,
    curvature_contrast=0.25)
plt.savefig(f'figures/Subject_{sub}_mllm_{metric}.png')


#### for LLM
mat_file_path = f'data/ROIs/Subject_{sub}/fmri_shared_1k_all_rois_3d_{metric}_chatgpt.mat'
mat_contents = loadmat(mat_file_path)
test_data_LLM=mat_contents['data']

mat_file_path = f'data/ROIs/Subject_{sub}/fmri_shared_1k_all_rois_3d_nc.mat'
mat_contents = loadmat(mat_file_path)
test_data_nc=mat_contents['data']
test_data_nc[test_data_nc < 20] = 100
test_data_LLM = 100 * test_data_LLM / test_data_nc

test_data = np.transpose(test_data_LLM, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_Human)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)
cortex.quickshow(vol_data, height=1024, dpi=100, with_labels=True, with_colorbar=True, with_rois=True,labelsize="50pt", with_curvature=True, curvature_brightness=0.5,
    curvature_contrast=0.25)
plt.savefig(f'figures/Subject_{sub}_llm_{metric}.png')

#### for CLIPvision (Spose)
mat_file_path = f'data/ROIs/Subject_{sub}/fmri_shared_1k_all_rois_3d_{metric}_CLIPvison_ViT_L14.mat'
mat_contents = loadmat(mat_file_path)
test_data_CLIPvision=mat_contents['data']

mat_file_path = f'data/ROIs/Subject_{sub}/fmri_shared_1k_all_rois_3d_nc.mat'
mat_contents = loadmat(mat_file_path)
test_data_nc=mat_contents['data']
test_data_nc[test_data_nc < 20] = 100
test_data_CLIPvision = 100 * test_data_CLIPvision / test_data_nc

test_data = np.transpose(test_data_CLIPvision, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_Human)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)
cortex.quickshow(vol_data, height=1024, dpi=100, with_labels=True, with_colorbar=True, with_rois=True,labelsize="50pt", with_curvature=True, curvature_brightness=0.5,
    curvature_contrast=0.25)
plt.savefig(f'figures/Subject_{sub}_CLIPvision_{metric}.png')

#### for CLIPtext (Spose)
mat_file_path = f'data/ROIs/Subject_{sub}/fmri_shared_1k_all_rois_3d_{metric}_CLIPtext_ViT_L14.mat'
mat_contents = loadmat(mat_file_path)
test_data_CLIPtext=mat_contents['data']

mat_file_path = f'data/ROIs/Subject_{sub}/fmri_shared_1k_all_rois_3d_nc.mat'
mat_contents = loadmat(mat_file_path)
test_data_nc=mat_contents['data']
test_data_nc[test_data_nc < 20] = 100
test_data_CLIPtext = 100 * test_data_CLIPtext / test_data_nc

test_data = np.transpose(test_data_CLIPtext, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_Human)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)
cortex.quickshow(vol_data, height=1024, dpi=100, with_labels=True, with_colorbar=True, with_rois=True,labelsize="50pt", with_curvature=True, curvature_brightness=0.5,
    curvature_contrast=0.25)
plt.savefig(f'figures/Subject_{sub}_CLIPtext_{metric}.png')