import os
from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = "cpu"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2Model.from_pretrained('gpt2')
model.to(device)

with open("data/variables/image_descriptions_meta_without_number_space.txt", "r") as file:
    lines = file.readlines()

text_features = []
max_length = 512

for i, line in enumerate(lines):
    print(i + 1)
    inputs = tokenizer([line], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state  # shape: (batch_size, sequence_length, hidden_size)

        embedding = last_hidden_states.mean(dim=1).cpu().numpy()  # shape: (batch_size, hidden_size)
        text_features.append(embedding.squeeze())  # shape: (hidden_size,)

text_feature_array = np.array(text_features)
print(text_feature_array.shape)
np.save('data/variables/things1854image_caption_feas_gpt2.npy', text_feature_array)
