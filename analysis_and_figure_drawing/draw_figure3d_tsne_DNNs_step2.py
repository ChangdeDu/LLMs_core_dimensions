
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

model_names_layer = ['feas_CLIP_ViT_L14', 'caption_feas_CLIP_ViT_L14', 'feas_simclr','feas_resnet18','feas_vgg16','feas_cornet_s','feas_alexnet','feas_gabor']
model_names_spose = ['CLIPvison_ViT_L14', 'CLIPtext_ViT_L14', 'simclr','resnet18','vgg16','cornet_s','alexnet','gabor']
title_names = ['CLIPvision', 'CLIPtext','SimCLR','ResNet18', 'VGG16','CORnet_S','AlexNet','Gabor']
for model in range(len(model_names_spose)):
    model_name = model_names_spose[model]
    title_name = title_names[model]
    data = loadmat(f'data/DNNs/{model_name}/spose_embedding_sorted_merge_tsne.mat')
    features = data['Ytsne']  
    
    concepts_path = 'data/things_concepts.tsv'
    concepts = pd.read_csv(concepts_path, delimiter='\t')
    
    categories = ['animal', 'vehicle', 'clothing', 'plant','food', 'furniture', 'container', 'tool','body part','weapon','decoration']
    colors = ['red', 'green', 'orange', 'blue', 'brown', 'purple', 'pink', 'yellow', 'turquoise', 'greenyellow', 'steelblue']
    
    c = np.zeros(features.shape[0])
    for i, category in enumerate(categories):
        subset = concepts[concepts["Top-down Category (WordNet)"] == category]
        c[subset.index] = i + 1
    
    
    X = features
    
    for i, category in enumerate(categories):
        plt.scatter(*zip(*X[c == i + 1]), c=colors[i], label=category, s=15)
    plt.scatter(*zip(*X[c == 0]), c='black', label='other', alpha=.1, s=15)
    plt.axis('off')
    plt.title(title_name+' (SPoSE)', fontsize=22)
    # plt.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.15))
    plt.savefig(f'figures/tsne_spose_{model_name}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    
for model in range(len(model_names_layer)):
    model_name = model_names_layer[model]
    title_name = title_names[model]
    data = loadmat(f'data/variables/things1854image_{model_name}_tsne.mat')
    features = data['Ytsne']  
    
    concepts_path = 'data/things_concepts.tsv'
    concepts = pd.read_csv(concepts_path, delimiter='\t')
    
    categories = ['animal', 'vehicle', 'clothing', 'plant','food', 'furniture', 'container', 'tool','body part','weapon','decoration']
    colors = ['red', 'green', 'orange', 'blue', 'brown', 'purple', 'pink', 'yellow', 'turquoise', 'greenyellow', 'steelblue']
    
    c = np.zeros(features.shape[0])
    for i, category in enumerate(categories):
        subset = concepts[concepts["Top-down Category (WordNet)"] == category]
        c[subset.index] = i + 1
    
    
    X = features
    
    for i, category in enumerate(categories):
        plt.scatter(*zip(*X[c == i + 1]), c=colors[i], label=category, s=15)
    plt.scatter(*zip(*X[c == 0]), c='black', label='other', alpha=.1, s=15)
    plt.axis('off')
    plt.title(title_name+' (Original)', fontsize=22)
    # plt.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.15))
    plt.savefig(f'figures/tsne_layer_{model_name}.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()

