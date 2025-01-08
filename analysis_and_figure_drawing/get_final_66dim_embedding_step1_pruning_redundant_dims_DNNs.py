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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# model_sets= ['alexnet','CLIPtext_ViT_L14','CLIPvison_ViT_L14','cornet_s','gabor','resnet18']
# lr_sets = [0.007,0.006,0.006,0.007,0.007,0.007]

model_sets= ['simclr','vgg16']
lr_sets = [0.007,0.007]

for i in range(len(model_sets)):
    model_name = model_sets[i]
    lr = lr_sets[i]
    basePath = parent_dir+f'/model_training/SPoSE/results/{model_name}/100d/{lr}/'
    seedIDs = ['seed42','seed142','seed242']

    print(f"model_name: {model_name}")
    mergedata = []
    for index, ID in enumerate(seedIDs):
        folder = basePath + ID + '/'
        data = np.load(folder + 'weights_sorted.npy')
        pruned_loc = data
        mergedata.append(data)

        # Generate the corresponding ID in s01, s02, etc
        s_index = '{:02d}'.format(index+1)
        new_ID = 's' + s_index

        save_folder = f'data/DNNs/{model_name}/reference_models_{model_name}_spose/' + new_ID
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        np.savetxt(save_folder + '/spose_embedding_sorted.txt', pruned_loc, fmt='%.8f')

    mergedata = np.hstack(mergedata)
    mergedata_without_redundancy, removed_features = remove_highly_correlated_features(mergedata, threshold=0.4)
    mergedata_without_redundancy = sort_dim(mergedata_without_redundancy)
    np.savetxt(f'data/DNNs/{model_name}/spose_embedding_sorted_merge.txt', mergedata_without_redundancy, fmt='%.8f')


