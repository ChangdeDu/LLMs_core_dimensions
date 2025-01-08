import numpy as np
from numpy.linalg import norm
import os
import shutil
from scipy.io import loadmat

def cosine_similarity(feat1, feat2):
    return np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))

subject_list =[1,2,5,7]
roi_sets = ['EarlyVis','RSC', 'OPA', 'EBA', 'FFA-1','FFA-2', 'PPA','IPS','AG','TPOJ1','Broca']
reference_sets=['trainset','validationset']

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for sub in subject_list:
	for roi in roi_sets:
		for setname in reference_sets:
			print(f"subject: {sub}, roi: {roi}, setname: {setname}")
			input_triplet_list_file = parent_dir+'/analysis_and_figure_drawing/data/ROIs/'+setname+'_random_generated.txt'
			savepath = parent_dir+'/analysis_and_figure_drawing/data/ROIs/Subject_' + str(sub)+'/triplet_dataset/'
			os.makedirs(savepath, exist_ok=True)
			output_triplet_list_file = savepath+setname+'.txt'
			mat_contents = loadmat(parent_dir+'/analysis_and_figure_drawing/data/ROIs/Subject_' + str(sub)+ '/fmri_shared_1k_'+roi +'.mat')
			features = mat_contents['data']

			with open(output_triplet_list_file, 'w') as output_file:
				cnt = 0
				with open(input_triplet_list_file, 'r') as file:
					triplets_cnt = sum(1 for line in file)
				with open(input_triplet_list_file, 'r') as file:
					for line in file:
						print(f'Process {cnt}/{triplets_cnt} triplets', end='\r')
						indices = list(map(int, line.strip().split()))

						feat1, feat2, feat3 = features[indices[0]], features[indices[1]], features[indices[2]]

						sim12 = cosine_similarity(feat1, feat2)
						sim13 = cosine_similarity(feat1, feat3)
						sim23 = cosine_similarity(feat2, feat3)

						if sim12 >= sim13 and sim12 >= sim23:
							ordered_indices = [indices[0], indices[1], indices[2]]
						elif sim13 >= sim12 and sim13 >= sim23:
							ordered_indices = [indices[0], indices[2], indices[1]]
						else:
							ordered_indices = [indices[1], indices[2], indices[0]]

						# print(f"Original: {indices}, Ordered: {ordered_indices}")
						# print(f"ooo is: {ordered_indices[-1]}")
						cnt += 1

						output_file.write(f"{ordered_indices[0]} {ordered_indices[1]} {ordered_indices[2]}\n")

			source_file = os.path.join(savepath, f'{setname}.txt')
			target_directory = parent_dir+'/model_training/SPoSE/data/NSD/Subject_' + str(sub)+'/'+roi
			os.makedirs(target_directory, exist_ok=True)
			if setname=='trainset':
				target_file = os.path.join(target_directory, f'train_90.txt')
				shutil.copyfile(source_file, target_file)
				print(f"File copied and renamed to {target_file}")
			elif setname=='validationset':
				target_file = os.path.join(target_directory, f'test_10.txt')
				shutil.copyfile(source_file, target_file)
				print(f"File copied and renamed to {target_file}")