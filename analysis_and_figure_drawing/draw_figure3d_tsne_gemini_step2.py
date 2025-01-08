
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


data = loadmat('data/MLLMs/Gemini_Pro_Vision/spose_embedding_66d_sorted_gemini_tsne.mat')
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
plt.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.15))
plt.savefig('tsne_gemini.pdf', bbox_inches='tight')
plt.show()



