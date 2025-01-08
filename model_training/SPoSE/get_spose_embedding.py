import subprocess

seed_sets =['42']
model_sets= ['vgg16']
lmbda = 0.007


for seed in seed_sets:
	for model in model_sets:
		print(f"model: {model}, seed: {seed}")
		cmd = f"python train.py --task odd_one_out --modality {model}/ --triplets_dir ./data/{model}_things/ --learning_rate 0.001 --lmbda {lmbda} --embed_dim 100 --batch_size 100 --epochs 500 --window_size 100 --steps 100 --sampling_method normal --device cpu --rnd_seed {seed}"
		subprocess.run(cmd, shell=True)
