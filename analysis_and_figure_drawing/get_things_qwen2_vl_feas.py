import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import time
import re
import csv
from tqdm import tqdm
import numpy as np

# Load model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/nfs/diskstation/DataStation/ChangdeDu/qwen2-vl-7B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
).to("cuda")

processor = AutoProcessor.from_pretrained("/nfs/diskstation/DataStation/ChangdeDu/qwen2-vl-7B")
print("=" * 64)

# Define paths
img_root = 'data/THINGS_visual_stimuli_1854/'
features = []

# Loop through images
for i in tqdm(range(1854)):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_root + f"image_{i + 1}_ori.jpg",
                },
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # Perform inference
    with torch.no_grad():  # Avoid tracking gradients to save memory
        pixel_values = inputs['pixel_values']
        image_grid_thw = inputs['image_grid_thw']
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)

        # Convert to float32 and detach from GPU memory
        image_embeds = image_embeds.to(torch.float32).cpu().detach().numpy()

    # Append mean feature
    features.append(np.mean(image_embeds, axis=0))

    # Manually free memory
    del inputs, pixel_values, image_grid_thw, image_embeds
    torch.cuda.empty_cache()

# Save features
Features = np.array(features)
print(Features.shape)
np.save(
    'data/variables/things1854image_feas_' + 'qwen2_vl_7B.npy',
    Features)
