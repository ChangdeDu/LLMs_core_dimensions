import numpy as np
import os
import pandas as pd

def remove_highly_correlated_features(data, threshold):
    df = pd.DataFrame(data)
    n_features = df.shape[1]
    selected_features = list(range(n_features))
    removed_features = []

    while True:
        redundant = False
        for i in range(n_features):
            if i in selected_features:
                correlation_with_other_features = df[selected_features].corrwith(df[i])
                correlated_features = correlation_with_other_features[correlation_with_other_features > threshold].index.tolist()

                if len(correlated_features) > 0:
                    removed_features.extend(correlated_features[1:])
                    for feature in correlated_features[1:]:
                        selected_features.remove(feature)
                        redundant = True

        if not redundant:
            break

    return df[selected_features].values, removed_features

def sort_dim(data):
    column_sum_indices = np.argsort(np.sum(data, axis=0))
    column_sum_indices = column_sum_indices[::-1]
    data = data[:, column_sum_indices]
    return data
    
basePath = 'data/MLLMs/Gemini_Pro_Vision/gemini_pro_vision_spose/0.0035/'
seedIDs = ['seed142','seed1142','seed2142','seed3142','seed4142','seed5142','seed6142','seed7142','seed8142','seed9142','seed10142','seed11142','seed12142','seed13142','seed14142','seed15142','seed16142','seed17142','seed18142','seed19142','seed20142','seed211142','seed311142','seed411142','seed511142','seed611142','seed711142','seed811142','seed911142']
              
mergedata = []   
for index, ID in enumerate(seedIDs):
    folder = basePath + ID + '/'
    data = np.load(folder + 'weights_sorted.npy')
    pruned_loc = data
    mergedata.append(data) 
    
    # Generate the corresponding ID in s01, s02, etc
    s_index = '{:02d}'.format(index+1)
    new_ID = 's' + s_index

    save_folder = 'data/MLLMs/Gemini_Pro_Vision/reference_models_gemini_spose/' + new_ID
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    np.savetxt(save_folder + '/spose_embedding_sorted.txt', pruned_loc, fmt='%.8f')
    
mergedata = np.hstack(mergedata)  
mergedata_without_redundancy, removed_features = remove_highly_correlated_features(mergedata, threshold=0.4)
mergedata_without_redundancy = sort_dim(mergedata_without_redundancy)
np.savetxt('data/MLLMs/Gemini_Pro_Vision/spose_embedding_sorted_merge.txt', mergedata_without_redundancy, fmt='%.8f')


