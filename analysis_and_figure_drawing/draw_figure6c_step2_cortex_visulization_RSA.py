import cortex
import numpy as np
from scipy.io import loadmat

np.random.seed(1234)

sub = 1 # 2,5,7
subject = 'subj0'+str(sub)+'_ChangdeDuNMI'
xfm = 'func1pt8_to_anat0pt8_autoFSbbr'

#### for Human
mat_file_path = 'data/ROIs/rsa_scores_unnormlized_Subject_'+str(sub)+'_Human.mat'
mat_contents = loadmat(mat_file_path)
test_data_Human=mat_contents['model_data']
test_data = np.transpose(test_data_Human, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_Human)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)
cortex.quickshow(vol_data, height=1024, dpi=100, with_labels=True, with_colorbar=True, with_rois=True,labelsize="50pt", with_curvature=True, curvature_brightness=0.5,
    curvature_contrast=0.25)
plt.savefig(f'figures/Subject_{sub}_humans_rsa.png')

#### for MLLM
mat_file_path = 'data/ROIs/rsa_scores_unnormlized_Subject_'+str(sub)+'_MLLM.mat'
mat_contents = loadmat(mat_file_path)
test_data_MLLM=mat_contents['model_data']
test_data = np.transpose(test_data_MLLM, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_Human)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)

#### for LLM
mat_file_path = 'data/ROIs/rsa_scores_unnormlized_Subject_'+str(sub)+'_LLM.mat'
mat_contents = loadmat(mat_file_path)
test_data_LLM=mat_contents['model_data']
test_data = np.transpose(test_data_LLM, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_Human)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)

#### for CLIPvision
mat_file_path = 'data/ROIs/rsa_scores_unnormlized_Subject_'+str(sub)+'_CLIPvision.mat'
mat_contents = loadmat(mat_file_path)
test_data_CLIPvision=mat_contents['model_data']
test_data = np.transpose(test_data_CLIPvision, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_Human)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)

#### for CLIPtext
mat_file_path = 'data/ROIs/rsa_scores_unnormlized_Subject_'+str(sub)+'_CLIPtext.mat'
mat_contents = loadmat(mat_file_path)
test_data_CLIPtext=mat_contents['model_data']
test_data = np.transpose(test_data_CLIPtext, (2, 1, 0))
test_data[test_data == 0] = np.nan
vol_data = cortex.Volume(test_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(test_data_Human)), cmap="gist_heat_r")
cortex.webshow(vol_data, recache=True)