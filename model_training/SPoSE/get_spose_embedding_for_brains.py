import subprocess

# seed_sets =['42','142','242','342','442','542','642','742','842','942','1042','1142','1242','1342','1442','1542','1642','1742','1842','1942']
seed_sets =['242']
subject_list =[7]
roi_sets = ['EBA']
# roi_sets = ['FFA-1','FFA-2', 'PPA','IPS','AG','TPOJ1','Broca']

lmbda = 0.008
for seed in seed_sets:
	for sub in subject_list:
		for roi in roi_sets:
			print(f"subject: {sub}, model: {roi}, seed: {seed}")
			cmd = f"python train.py --task odd_one_out --modality {'Subject_' + str(sub)+'/'+roi}/ --triplets_dir ./data/NSD/{'Subject_' + str(sub)+'/'+roi}/ --learning_rate 0.001 --lmbda {lmbda} --embed_dim 100 --batch_size 100 --epochs 500 --window_size 100 --steps 100 --sampling_method normal --device cpu --rnd_seed {seed}"
			subprocess.run(cmd, shell=True)
