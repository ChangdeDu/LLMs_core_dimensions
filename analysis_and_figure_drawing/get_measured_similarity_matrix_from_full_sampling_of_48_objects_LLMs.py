import numpy as np
import scipy.io as sio

model_sets = ['Llama3.1','Qwen2_VL']
num_objects = 1854
for model in model_sets:
    print("model name:", model)
    if model=='Llama3.1':
        prefix = 'data/LLMs/'
        filename = '/triplet_dataset/full_sampling_for_48_objects_Llama3.1_8b.txt'
    elif model=='Qwen2_VL':
        prefix = 'data/MLLMs/'
        filename = '/triplet_dataset/full_sampling_for_48_objects_qwen2_vl.txt'

    with open(prefix + model + filename, 'r') as file:
        lines = file.readlines()

    mid_index = len(lines) // 2
    lines_first_half = lines[:mid_index]
    lines_second_half = lines[mid_index:]

    # ##############################################
    # ## full data
    # ##############################################
    print("full data")
    trial_count_full = np.zeros((num_objects, num_objects), dtype=int)
    combination_count_full = np.zeros((num_objects, num_objects), dtype=int)

    # First pass to fill trial_count and combination_count
    for line in lines:
        img1, img2, img3 = map(int, line.strip().split())
        trial_count_full[img1, img2] += 1
        trial_count_full[img2, img1] += 1
        combination_count_full[img1, img2] += 1
        combination_count_full[img2, img1] += 1
        combination_count_full[img1, img3] += 1
        combination_count_full[img3, img1] += 1
        combination_count_full[img2, img3] += 1
        combination_count_full[img3, img2] += 1

    similarity_matrix_full = np.zeros((num_objects, num_objects), dtype=float)
    # Compute similarity matrix
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            if combination_count_full[i, j] > 0:
                numerator = trial_count_full[i, j]
                similarity_score = numerator / combination_count_full[i, j]
                similarity_matrix_full[i, j] = similarity_score
                similarity_matrix_full[j, i] = similarity_score

    np.fill_diagonal(similarity_matrix_full, 1)

    ##############################################
    ## first_half data
    ##############################################
    print("first_half data")
    trial_count_first_half = np.zeros((num_objects, num_objects), dtype=int)
    combination_count_first_half = np.zeros((num_objects, num_objects), dtype=int)

    # First pass to fill trial_count and combination_count
    for line in lines_first_half:
        img1, img2, img3 = map(int, line.strip().split())
        trial_count_first_half[img1, img2] += 1
        trial_count_first_half[img2, img1] += 1
        combination_count_first_half[img1, img2] += 1
        combination_count_first_half[img2, img1] += 1
        combination_count_first_half[img1, img3] += 1
        combination_count_first_half[img3, img1] += 1
        combination_count_first_half[img2, img3] += 1
        combination_count_first_half[img3, img2] += 1

    similarity_matrix_first_half = np.zeros((num_objects, num_objects), dtype=float)
    # Compute similarity matrix
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            if combination_count_first_half[i, j] > 0:
                numerator = trial_count_first_half[i, j]
                similarity_score = numerator / combination_count_first_half[i, j]
                similarity_matrix_first_half[i, j] = similarity_score
                similarity_matrix_first_half[j, i] = similarity_score

    np.fill_diagonal(similarity_matrix_first_half, 1)

    ##############################################
    ## second_half data
    ##############################################
    print("second_half data")
    trial_count_second_half = np.zeros((num_objects, num_objects), dtype=int)
    combination_count_second_half = np.zeros((num_objects, num_objects), dtype=int)

    # First pass to fill trial_count and combination_count
    for line in lines_second_half:
        img1, img2, img3 = map(int, line.strip().split())
        trial_count_second_half[img1, img2] += 1
        trial_count_second_half[img2, img1] += 1
        combination_count_second_half[img1, img2] += 1
        combination_count_second_half[img2, img1] += 1
        combination_count_second_half[img1, img3] += 1
        combination_count_second_half[img3, img1] += 1
        combination_count_second_half[img2, img3] += 1
        combination_count_second_half[img3, img2] += 1

    similarity_matrix_second_half = np.zeros((num_objects, num_objects), dtype=float)
    # Compute similarity matrix
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            if combination_count_second_half[i, j] > 0:
                numerator = trial_count_second_half[i, j]
                similarity_score = numerator / combination_count_second_half[i, j]
                similarity_matrix_second_half[i, j] = similarity_score
                similarity_matrix_second_half[j, i] = similarity_score

    np.fill_diagonal(similarity_matrix_second_half, 1)

    sio.savemat(prefix + model + '/RSM1854_get_from_full_sampling_of_48_objects_' + model + '.mat',
                {'RSM1854_triplet': similarity_matrix_full})
    sio.savemat(prefix + model + '/RSM1854_get_from_full_sampling_of_48_objects_' + model + '_splithalf.mat',
                {'RSM1854_triplet_split1': similarity_matrix_first_half,
                 'RSM1854_triplet_split2': similarity_matrix_second_half})