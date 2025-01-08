import numpy as np
from himalaya.ridge import RidgeCV

tol = 3
alphas = np.logspace(-tol, tol, 100)
# ###################################################################################
# ## using multimodal features as input for linear regression, chatgpt as target
# ###################################################################################
# X_image = np.load('data/variables/things1854image_feas_CLIP_ViT_L14.npy')
# X_text = np.load('data/variables/things1854image_caption_feas_CLIP_ViT_L14.npy')
# X = np.concatenate((X_image, X_text), axis=1)
# Y = np.loadtxt('data/LLMs/ChatGPT-3.5/spose_embedding_66d_sorted_chatgpt.txt')
# model = RidgeCV(alphas=alphas, fit_intercept=True, solver='svd', solver_params=None, cv=5)
# model.fit(X, Y)
# print(model.best_alphas_)
#
# # infer the 66 embedding for nsd_shared_1k
# X_test_image = np.load('data/variables/shared_1k_image_feas_CLIP_ViT_L14.npy')
# X_test_text = np.load('data/variables/shared_1k_image_caption_feas_CLIP_ViT_L14.npy')
# X_test = np.concatenate((X_test_image, X_test_text), axis=1)
# nsd_shared_1k_predict = model.predict(X_test)
# np.savetxt('data/LLMs/ChatGPT-3.5/spose_embedding_nsd_shared_1k_predicted_from_chatgpt.txt',
#            nsd_shared_1k_predict, fmt='%.8f')
#
# # infer the 66 embedding for nsd_trn
# for subjid in [1, 2, 5, 7]:
#     X_test_image = np.load('data/variables/NSD_trn_image_feas_CLIP_ViT_L14'+ '_subj0' + str(subjid) + '.npy')
#     X_test_text = np.load('data/variables/NSD_trn_image_caption_feas_CLIP_ViT_L14'+ '_subj0' + str(subjid) + '.npy')
#     X_test = np.concatenate((X_test_image, X_test_text), axis=1)
#     nsd_trn_predict = model.predict(X_test)
#     np.savetxt('data/LLMs/ChatGPT-3.5/spose_embedding_nsd_trn_subj0' + str(subjid)+'_predicted_from_chatgpt.txt', nsd_trn_predict, fmt='%.8f')
#
# ###################################################################################
# ## using multimodal features as input for linear regression, gemini as target
# ###################################################################################
# X_image = np.load('data/variables/things1854image_feas_CLIP_ViT_L14.npy')
# X_text = np.load('data/variables/things1854image_caption_feas_CLIP_ViT_L14.npy')
# X = np.concatenate((X_image, X_text), axis=1)
# Y = np.loadtxt('data/MLLMs/Gemini_Pro_Vision/spose_embedding_66d_sorted_gemini.txt')
# model = RidgeCV(alphas=alphas, fit_intercept=True, solver='svd', solver_params=None, cv=5)
# model.fit(X, Y)
# print(model.best_alphas_)
#
# # infer the 66 embedding for nsd_shared_1k
# X_test_image = np.load('data/variables/shared_1k_image_feas_CLIP_ViT_L14.npy')
# X_test_text = np.load('data/variables/shared_1k_image_caption_feas_CLIP_ViT_L14.npy')
# X_test = np.concatenate((X_test_image, X_test_text), axis=1)
# nsd_shared_1k_predict = model.predict(X_test)
# np.savetxt('data/MLLMs/Gemini_Pro_Vision/spose_embedding_nsd_shared_1k_predicted_from_gemini.txt',
#            nsd_shared_1k_predict, fmt='%.8f')
#
# # infer the 66 embedding for nsd_trn
# for subjid in [1, 2, 5, 7]:
#     X_test_image = np.load('data/variables/NSD_trn_image_feas_CLIP_ViT_L14'+ '_subj0' + str(subjid) + '.npy')
#     X_test_text = np.load('data/variables/NSD_trn_image_caption_feas_CLIP_ViT_L14'+ '_subj0' + str(subjid) + '.npy')
#     X_test = np.concatenate((X_test_image, X_test_text), axis=1)
#     nsd_trn_predict = model.predict(X_test)
#     np.savetxt('data/MLLMs/Gemini_Pro_Vision/spose_embedding_nsd_trn_subj0' + str(subjid)+'_predicted_from_gemini.txt', nsd_trn_predict, fmt='%.8f')
#
#
# ###################################################################################
# ## using multimodal features as input for linear regression, human as target
# ###################################################################################
# X_image = np.load('data/variables/things1854image_feas_CLIP_ViT_L14.npy')
# X_text = np.load('data/variables/things1854image_caption_feas_CLIP_ViT_L14.npy')
# X = np.concatenate((X_image, X_text), axis=1)
# Y = np.loadtxt('data/Humans/spose_embedding_66d_sorted_humans.txt')
# model = RidgeCV(alphas=alphas, fit_intercept=True, solver='svd', solver_params=None, cv=5)
# model.fit(X, Y)
# print(model.best_alphas_)
#
# # infer the 66 embedding for nsd_shared_1k
# X_test_image = np.load('data/variables/shared_1k_image_feas_CLIP_ViT_L14.npy')
# X_test_text = np.load('data/variables/shared_1k_image_caption_feas_CLIP_ViT_L14.npy')
# X_test = np.concatenate((X_test_image, X_test_text), axis=1)
# nsd_shared_1k_predict = model.predict(X_test)
# np.savetxt('data/Humans/spose_embedding_nsd_shared_1k_predicted_from_humans.txt',
#            nsd_shared_1k_predict, fmt='%.8f')
#
# # infer the 66 embedding for nsd_trn
# for subjid in [1, 2, 5, 7]:
#     X_test_image = np.load('data/variables/NSD_trn_image_feas_CLIP_ViT_L14'+ '_subj0' + str(subjid) + '.npy')
#     X_test_text = np.load('data/variables/NSD_trn_image_caption_feas_CLIP_ViT_L14'+ '_subj0' + str(subjid) + '.npy')
#     X_test = np.concatenate((X_test_image, X_test_text), axis=1)
#     nsd_trn_predict = model.predict(X_test)
#     np.savetxt('data/Humans/spose_embedding_nsd_trn_subj0' + str(subjid)+'_predicted_from_humans.txt', nsd_trn_predict, fmt='%.8f')
#

###################################################################################
## using multimodal features as input for linear regression, the spose embedding of DNNs as target
###################################################################################
model_sets = ['CLIPvison_ViT_L14', 'CLIPtext_ViT_L14', 'resnet18','vgg16','cornet_s','alexnet','gabor']
for model_name in model_sets:
    X_image = np.load('data/variables/things1854image_feas_CLIP_ViT_L14.npy')
    X_text = np.load('data/variables/things1854image_caption_feas_CLIP_ViT_L14.npy')
    X = np.concatenate((X_image, X_text), axis=1)
    Y = np.loadtxt(f'data/DNNs/{model_name}/spose_embedding_sorted_merge.txt')
    model = RidgeCV(alphas=alphas, fit_intercept=True, solver='svd', solver_params=None, cv=5)
    model.fit(X, Y)
    print(model.best_alphas_)

    # infer the 66 embedding for nsd_shared_1k
    X_test_image = np.load('data/variables/shared_1k_image_feas_CLIP_ViT_L14.npy')
    X_test_text = np.load('data/variables/shared_1k_image_caption_feas_CLIP_ViT_L14.npy')
    X_test = np.concatenate((X_test_image, X_test_text), axis=1)
    nsd_shared_1k_predict = model.predict(X_test)
    np.savetxt(f'data/DNNs/{model_name}/spose_embedding_nsd_shared_1k_predicted_from_{model_name}.txt',
               nsd_shared_1k_predict, fmt='%.8f')

    # infer the 66 embedding for nsd_trn
    for subjid in [1, 2, 5, 7]:
        X_test_image = np.load('data/variables/NSD_trn_image_feas_CLIP_ViT_L14'+ '_subj0' + str(subjid) + '.npy')
        X_test_text = np.load('data/variables/NSD_trn_image_caption_feas_CLIP_ViT_L14'+ '_subj0' + str(subjid) + '.npy')
        X_test = np.concatenate((X_test_image, X_test_text), axis=1)
        nsd_trn_predict = model.predict(X_test)
        np.savetxt(f'data/DNNs/{model_name}/spose_embedding_nsd_trn_subj0{subjid}_predicted_from_{model_name}.txt', nsd_trn_predict, fmt='%.8f')
