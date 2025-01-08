import torch
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
from scipy import io
import torchvision.models as models

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

all_models = torchvision.models.__dict__.keys()
print(all_models)

model_sets= ['vgg16','resnet18','alexnet']
for modelname in model_sets:
    if modelname == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif modelname == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif modelname == 'resnet18':
        model = models.resnet18(pretrained=True)

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
        feature_flatten = feature.cpu().flatten().numpy()
        features.append(feature_flatten)
        print(f'Image: {i + 1}, Pooled shape: {feature_flatten.shape}')
        image_feature_array = np.array(features)
        print(f'feature_array shape: {image_feature_array.shape}')
    np.save('data/variables/things1854image_feas_'+modelname+'.npy',np.squeeze(image_feature_array))