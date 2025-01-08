import os
import re
import torch
import transformers
import numpy as np
from transformers import AutoTokenizer, LlamaModel


def get_sentence_embedding(sentences, tokenizer, model, max_length=128, device='cuda' if torch.cuda.is_available() else 'cpu', pool='average'):
    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # 使用最后一层的平均值作为句子 embedding
    last_hidden_state = outputs.last_hidden_state
    attention_mask = inputs.attention_mask  # (batch_size, sequence_length)
    if pool == 'average':
        # 排除 padding tokens，进行加权平均
        # 扩展 attention_mask 的维度，使其与 last_hidden_state 匹配
        attention_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)

        # 使用 attention_mask 对 last_hidden_state 加权
        weighted_hidden_state = last_hidden_state * attention_mask

        # 对有效部分进行池化（如求平均）
        embeddings = weighted_hidden_state.sum(dim=1) / attention_mask.sum(dim=1)
    elif pool == 'max':
        valid_tokens_idx = attention_mask.bool()  # 创建一个布尔 mask，指示有效 token
        valid_hidden_state = last_hidden_state[valid_tokens_idx]  # 选择有效 token 的表示

        # 对有效 token 进行最大池化
        embeddings = valid_hidden_state.max(dim=0).values
    elif pool == 'last':
        valid_tokens_idx = attention_mask[0].nonzero(as_tuple=True)[0]  # 获取有效 token 的索引
        last_valid_token_idx = valid_tokens_idx[-1]  # 获取最后一个有效 token 的索引

        # 获取最后一个有效 token 的表示
        embeddings = last_hidden_state[0, last_valid_token_idx, :]
    else:
        raise ValueError()
    return embeddings


if __name__ == '__main__':


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = "/data1/home/data/weights/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = LlamaModel.from_pretrained(model_id).to(device)
    # 设置 pad_token 为 eos_token
    tokenizer.pad_token = tokenizer.eos_token

    ## for name+defination
    with open("data/variables/image_descriptions_meta_without_number_space.txt", "r") as file:
        lines = file.readlines()

    text_features = []
    max_length = 512

    for i, line in enumerate(lines):
        print(i + 1)

        embeddings = get_sentence_embedding(line, tokenizer, model, max_length=max_length)
        embeddings = embeddings.cpu().numpy()
        text_features.append(embeddings.squeeze())

    text_feature_array = np.array(text_features)
    print(text_feature_array.shape)
    np.save('data/variables/things1854image_caption_feas_LLama3-1-8B.npy', text_feature_array)


    ## for image caption
    with open("data/variables/image_descriptions_llava_without_number_space.txt", "r") as file:
        lines = file.readlines()

    text_features = []
    max_length = 512

    for i, line in enumerate(lines):
        print(i + 1)

        embeddings = get_sentence_embedding(line, tokenizer, model, max_length=max_length)
        embeddings = embeddings.cpu().numpy()
        text_features.append(embeddings.squeeze())

    text_feature_array = np.array(text_features)
    print(text_feature_array.shape)
    np.save('data/variables/things1854image_caption_feas_LLama3-1-8B_llava_captions.npy', text_feature_array)

