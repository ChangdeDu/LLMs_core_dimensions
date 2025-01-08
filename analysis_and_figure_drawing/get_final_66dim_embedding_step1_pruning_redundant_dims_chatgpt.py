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
    
basePath = 'data/LLMs/ChatGPT-3.5/chatgpt_spose/0.004/'
seedIDs = ['seed642','seed142','seed242','seed342','seed442','seed542','seed742','seed842','seed942','seed1042','seed1142','seed1242','seed1342','seed1442','seed1542','seed1642','seed1742','seed1842','seed1942','seed2042']
       
mergedata = []   
for index, ID in enumerate(seedIDs):
    folder = basePath + ID + '/'
    data = np.load(folder + 'weights_sorted.npy')
    pruned_loc = data
    mergedata.append(data) 
    
    # Generate the corresponding ID in s01, s02, etc
    s_index = '{:02d}'.format(index+1)
    new_ID = 's' + s_index

    save_folder = 'data/LLMs/ChatGPT-3.5/reference_models_chatgpt_spose/' + new_ID
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    np.savetxt(save_folder + '/spose_embedding_sorted.txt', pruned_loc, fmt='%.8f')
    
mergedata = np.hstack(mergedata)  
mergedata_without_redundancy, removed_features = remove_highly_correlated_features(mergedata, threshold=0.4)
mergedata_without_redundancy = sort_dim(mergedata_without_redundancy)
np.savetxt('data/LLMs/ChatGPT-3.5/spose_embedding_sorted_merge.txt', mergedata_without_redundancy, fmt='%.8f')


