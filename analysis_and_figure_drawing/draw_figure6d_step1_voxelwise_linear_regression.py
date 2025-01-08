import numpy as np
from himalaya.ridge import RidgeCV
from scipy.stats import pearsonr
import os
from tqdm import tqdm
from statsmodels.stats.multitest import fdrcorrection
from scipy import io

def r2_score(Real, Pred):
    SSres = np.mean((Real - Pred) ** 2, 0)
    SStot = np.var(Real, 0, ddof=0)
    r_squared = 1 - SSres / SStot
    r_squared[np.isnan(r_squared)] = 0
    r_squared[r_squared < 0] = 1e-6
    return r_squared

def bootstrap_sampling(model, X_test, y_test, repeat, seed):
    np.random.seed(seed)
    rsq_dist = list()
    label_idx = np.arange(X_test.shape[0])
    yhat = model.predict(X_test)
    for _ in tqdm(range(repeat)):
        sampled_idx = np.random.choice(label_idx, replace=True, size=len(label_idx))
        y_test_sampled = y_test[sampled_idx, :]
        rsqs = r2_score(y_test_sampled, yhat)
        rsq_dist.append(rsqs)
    return rsq_dist

def fdr_correct_p(var):
    var = np.array(var)
    n = var.shape[0]
    p_vals = np.sum(var < 0, axis=0) / n  # proportions of permutation below 0
    fdr_p = fdrcorrection(p_vals)  # corrected p
    return fdr_p[1]

def ncsnr_to_nc(ncsnr):
    n=3 #for subject 1,2,5,7
    noise_ceiling = 100 * (ncsnr ** 2 / (ncsnr ** 2 + 1 / n))
    return noise_ceiling

model_list = ['chatgpt','gemini','humans','CLIP_vision','CLIP_text']
model_list = ['CLIPvison_ViT_L14', 'CLIPtext_ViT_L14', 'resnet18','vgg16','cornet_s','alexnet','gabor']
weight_saving_list = ['CLIPvison_ViT_L14', 'CLIPtext_ViT_L14', 'resnet18','vgg16','cornet_s','alexnet','gabor']

subject_list = [1,2,5,7]
roi_list = ['V1', 'V2', 'V3','V3ab','hV4','VO', 'PHC','LO','MT','MST','IPS','other']
repeat = 2000
tol = 3
alphas = np.logspace(-tol, tol, 100)

for subjid in subject_list:
    basepath = 'data/NSD_test_preprocessed/func1pt8mm/betas_fithrf_GLMdenoise_RR/subj0' + str(subjid) + '/'
    savepath = 'data/ROIs/Subject_' + str(subjid)
    os.makedirs(savepath, exist_ok=True)
    savepath = savepath + '/'
    pick_list = np.load('data/ROIs/corresponding_NSD_test_sample_index_Subject_' + str(subjid) + '.npy')
    pick_list_1 = np.load('data/ROIs/corresponding_NSD_test_sample_index_Subject_' + str(subjid) + '_trial_1.npy')
    pick_list_2 = np.load('data/ROIs/corresponding_NSD_test_sample_index_Subject_' + str(subjid) + '_trial_2.npy')
    pick_list_3 = np.load('data/ROIs/corresponding_NSD_test_sample_index_Subject_' + str(subjid) + '_trial_3.npy')
    roi_mask = np.load(basepath + 'roi_mask_V1.npy')
    roi_mask = np.transpose(roi_mask, (2, 1, 0)) # zyx-->xyz

    for model_name in model_list:
        if model_name=='chatgpt':
            X = np.loadtxt('data/LLMs/ChatGPT-3.5/spose_embedding_nsd_trn_subj0' + str(subjid) + '_predicted_from_chatgpt.txt')
            X_test = np.loadtxt('data/LLMs/ChatGPT-3.5/spose_embedding_nsd_shared_1k_predicted_from_chatgpt.txt')
        elif model_name == 'gemini':
            X = np.loadtxt('data/MLLMs/Gemini_Pro_Vision/spose_embedding_nsd_trn_subj0' + str(subjid) + '_predicted_from_gemini.txt')
            X_test = np.loadtxt('data/MLLMs/Gemini_Pro_Vision/spose_embedding_nsd_shared_1k_predicted_from_gemini.txt')
        elif model_name == 'humans':
            X = np.loadtxt('data/Humans/spose_embedding_nsd_trn_subj0' + str(subjid) + '_predicted_from_humans.txt')
            X_test = np.loadtxt('data/Humans/spose_embedding_nsd_shared_1k_predicted_from_humans.txt')
        elif model_name == 'CLIP_vision':
            X = np.load('data/variables/NSD_trn_image_feas_CLIP_ViT_L14' + '_subj0' + str(subjid) + '.npy')
            X_test = np.load('data/variables/shared_1k_image_feas_CLIP_ViT_L14.npy')
        elif model_name == 'CLIP_text':
            X = np.load('data/variables/NSD_trn_image_caption_feas_CLIP_ViT_L14' + '_subj0' + str(subjid) + '.npy')
            X_test = np.load('data/variables/shared_1k_image_caption_feas_CLIP_ViT_L14.npy')
        else:
            X = np.loadtxt(f'data/DNNs/{model_name}/spose_embedding_nsd_trn_subj0{subjid}_predicted_from_{model_name}.txt')
            X_test = np.loadtxt(f'data/DNNs/{model_name}/spose_embedding_nsd_shared_1k_predicted_from_{model_name}.txt')

        # build 3D array
        three_d_array_pearson_r = np.zeros(roi_mask.shape)
        three_d_array_pearson_r_normlized = np.zeros(roi_mask.shape)
        three_d_array_pearson_r_noisecelling = np.zeros(roi_mask.shape)
        three_d_array_r_sqs = np.zeros(roi_mask.shape)
        three_d_array_pearson_r_sqs_noisecelling = np.zeros(roi_mask.shape)
        three_d_array_fdr_p = np.zeros(roi_mask.shape)
        three_d_array_ncsnr = np.zeros(roi_mask.shape)
        three_d_array_nc = np.zeros(roi_mask.shape)
        # build 4D array
        four_d_array_weights = np.zeros((X_test.shape[1],) + roi_mask.shape)
        for roi in roi_list:
            print(f'Subject: {subjid},  Embedding name: {model_name},  ROI: {roi}')
            Y = np.load(basepath + 'trn_voxel_data_'+roi+'.npy')
            model = RidgeCV(alphas=alphas, fit_intercept=True, solver='svd', solver_params=None, cv=5)
            model.fit(X, Y)
            # print(model.best_alphas_)
            shared_1k_predict = model.predict(X_test)

            fmri = np.load(basepath + 'val_voxel_single_trial_data_' + roi + '.npy')
            fmri_shared_1k_1 = fmri[pick_list_1, :]
            fmri_shared_1k_2 = fmri[pick_list_2, :]
            fmri_shared_1k_3 = fmri[pick_list_3, :]
            fmri = np.load(basepath + 'val_voxel_multi_trial_data_' + roi + '.npy')
            fmri_shared_1k = fmri[pick_list, :]
            fmri_shared_1k_mean = (fmri_shared_1k_1 + fmri_shared_1k_2 + fmri_shared_1k_3)/3;

            # calculate r
            pearson_r1 = [pearsonr(fmri_shared_1k_mean[:, i], fmri_shared_1k_1[:, i]) for i in range(fmri_shared_1k_mean.shape[1])]
            pearson_r2 = [pearsonr(fmri_shared_1k_mean[:, i], fmri_shared_1k_2[:, i]) for i in range(fmri_shared_1k_mean.shape[1])]
            pearson_r3 = [pearsonr(fmri_shared_1k_mean[:, i], fmri_shared_1k_3[:, i]) for i in range(fmri_shared_1k_mean.shape[1])]
            noise_ceiling_r = [max(a.correlation, b.correlation, c.correlation) for a, b, c in zip(pearson_r1, pearson_r2, pearson_r3)]
            pearson_r = [pearsonr(fmri_shared_1k[:, i], shared_1k_predict[:, i]) for i in range(fmri_shared_1k.shape[1])]
            pearson_r = [(r.correlation if r.correlation >= 0 else 1e-6) for r in pearson_r]
            pearson_r_normlized = [a / b for a, b in zip(pearson_r, noise_ceiling_r) if b != 0]

            # calculate R2
            r_sqs1 = r2_score(fmri_shared_1k_mean, fmri_shared_1k_1)
            r_sqs2 = r2_score(fmri_shared_1k_mean, fmri_shared_1k_2)
            r_sqs3 = r2_score(fmri_shared_1k_mean, fmri_shared_1k_3)
            noise_ceiling_r_sqs = [max(element) for element in zip(r_sqs1, r_sqs2, r_sqs3)]
            r_sqs = r2_score(fmri_shared_1k, shared_1k_predict) #/ noise_ceiling_r_sqs

            # FDR correction
            r_sqs_dists = bootstrap_sampling(model, X_test, fmri_shared_1k, repeat=repeat, seed=36)
            fdr_p = fdr_correct_p(r_sqs_dists)
            # print('p<0.05=', np.sum(fdr_p < 0.05))

            roi_mask = np.load(basepath + 'roi_mask_' + roi + '.npy')
            roi_mask = np.transpose(roi_mask, (2, 1, 0))  # zyx-->xyz
            roi_ijk = np.argwhere(roi_mask != 0)
            # print('voxel_num=', roi_ijk.shape[0])

            # load ncsnr
            ncsnr = np.load(basepath + 'voxel_ncsnr_' + roi + '.npy')

            weights = model.coef_
            for voxel in range(fmri_shared_1k.shape[1]):
                i,j,k = roi_ijk[voxel,:]
                three_d_array_pearson_r[i,j,k]=pearson_r[voxel]
                three_d_array_pearson_r_normlized[i, j, k] = pearson_r_normlized[voxel]
                three_d_array_pearson_r_noisecelling[i, j, k] = noise_ceiling_r[voxel]
                three_d_array_pearson_r_sqs_noisecelling[i, j, k] = noise_ceiling_r_sqs[voxel]
                three_d_array_ncsnr[i, j, k] = ncsnr[voxel]
                three_d_array_nc[i, j, k] = ncsnr_to_nc(ncsnr[voxel])
                three_d_array_r_sqs[i, j, k] = r_sqs[voxel]
                three_d_array_fdr_p[i, j, k] = fdr_p[voxel]
                for fea in range(X_test.shape[1]):
                    four_d_array_weights[fea, i, j, k] = weights[fea,voxel]

        io.savemat(savepath + f"fmri_shared_1k_all_rois_3d_r_{model_name}.mat", {'data': three_d_array_pearson_r}) # in xyz format
        io.savemat(savepath + f"fmri_shared_1k_all_rois_3d_r_normlized_{model_name}.mat", {'data': three_d_array_pearson_r_normlized}) # in xyz format
        io.savemat(savepath + f"fmri_shared_1k_all_rois_3d_r_noisecelling_{model_name}.mat", {'data': three_d_array_pearson_r_noisecelling}) # in xyz format
        io.savemat(savepath + f"fmri_shared_1k_all_rois_3d_R2_noisecelling_{model_name}.mat", {'data': three_d_array_pearson_r_sqs_noisecelling})  # in xyz format
        io.savemat(savepath + f"fmri_shared_1k_all_rois_3d_R2_{model_name}.mat", {'data': three_d_array_r_sqs})
        io.savemat(savepath + f"fmri_shared_1k_all_rois_3d_fdr_p_{model_name}.mat", {'data': three_d_array_fdr_p})
        io.savemat(savepath + f"fmri_shared_1k_all_rois_3d_ncsnr.mat", {'data': three_d_array_ncsnr})
        io.savemat(savepath + f"fmri_shared_1k_all_rois_3d_nc.mat", {'data': three_d_array_nc})
        if model_name in weight_saving_list:
            io.savemat(savepath + f"all_rois_4d_weights_{model_name}.mat", {'data': four_d_array_weights})

