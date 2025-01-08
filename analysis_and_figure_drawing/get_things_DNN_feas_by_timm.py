import timm
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
avail_pretrained_models = timm.list_models(pretrained=True)
print(avail_pretrained_models)
model_sets= ['densenet121','vit_large_patch16_224','swin_large_patch4_window7_224']
for modelname in model_sets:
    model = timm.create_model(modelname, pretrained=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    features = []
    for i in range(1854):
        image_path = f"data/THINGS_visual_stimuli_1854/image_{i+1}_ori.jpg"
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = feature_extractor(image_tensor)
            # feature = torch.mean(feature, dim=1)
        feature_flatten = feature.cpu().flatten().numpy()
        features.append(feature_flatten)
        print(f'Image: {i + 1}, Pooled shape: {feature_flatten.shape}')
        image_feature_array = np.array(features)
        print(f'feature_array shape: {image_feature_array.shape}')
    np.save('data/variables/things1854image_feas_'+modelname+'.npy',np.squeeze(image_feature_array))