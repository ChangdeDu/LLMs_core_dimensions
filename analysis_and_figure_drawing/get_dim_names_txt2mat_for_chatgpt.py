import numpy as np
import scipy.io as sio

mat_data = []
for i in range(10):
    txt_filename = 'data/LLMs/ChatGPT-3.5/dim_naming_chatgpt/dimension_naming_subject_%s.txt' %(i+1)
    with open(txt_filename, 'r') as file:
        lines = file.readlines()
        row_data = []
        for line in lines:
            row_data.append(line)
        mat_data.append(row_data)

mat_data = np.array(mat_data, dtype=object)
sio.savemat('data/LLMs/ChatGPT-3.5/dimlabel_answers_chatgpt.mat', {'dimlabel_answers_chatgpt': mat_data}, format='5', appendmat=False)

    