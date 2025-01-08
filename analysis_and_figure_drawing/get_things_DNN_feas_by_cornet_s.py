import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import pandas
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision
import src.cornet
from PIL import Image

Image.warnings.simplefilter('ignore')

np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.benchmark = True
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

parser = argparse.ArgumentParser(description='CORnet')
parser.add_argument('--model', choices=['Z', 'R', 'RT', 'S'], default='S',
                    help='which model to train')

FLAGS, FIRE_FLAGS = parser.parse_known_args()


def get_model(pretrained=False):
    model = getattr(src.cornet, f'cornet_{FLAGS.model.lower()}')
    if FLAGS.model.lower() == 'r':
        model = model(pretrained=pretrained, times=5)
    else:
        model = model(pretrained=pretrained)
    return model

def test(image_path, layer='decoder', sublayer='avgpool', time_step=0, imsize=224):
    """
    Suitable for small image sets. If you have thousands of images or it is
    taking too long to extract features, consider using
    `torchvision.datasets.ImageFolder`, using `ImageNetVal` as an example.

    Kwargs:
        - layers (choose from: V1, V2, V4, IT, decoder)
        - sublayer (e.g., output, conv1, avgpool)
        - time_step (which time step to use for storing features)
        - imsize (resize image to how many pixels, default: 224)
    """
    model = get_model(pretrained=True)
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((imsize, imsize)),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        output = output.cpu().flatten().numpy()
        _model_feats.append(output)

    try:
        m = model.module
    except:
        m = model
    model_layer = getattr(getattr(m, layer), sublayer)
    model_layer.register_forward_hook(_store_feats)

    with torch.no_grad():
        model_feats = []
        try:
            im = Image.open(image_path).convert('RGB')
        except:
            raise FileNotFoundError(f'Unable to load {fname}')
        im = transform(im)
        im = im.unsqueeze(0).to(device)  # adding extra dimension for batch size of 1
        _model_feats = []
        model(im)
        model_feats.append(_model_feats[time_step])
        model_feats = np.concatenate(model_feats)
        return model_feats


model_sets= [f'cornet_{FLAGS.model.lower()}']
for modelname in model_sets:
    features = []
    for i in range(1854):
        image_path = f"data/THINGS_visual_stimuli_1854/image_{i+1}_ori.jpg"
        feature_flatten = test(image_path, layer='IT', sublayer='output', time_step=0, imsize=224)
        features.append(feature_flatten)
        print(f'Image: {i + 1}, Pooled shape: {feature_flatten.shape}')
        image_feature_array = np.array(features)
        print(f'feature_array shape: {image_feature_array.shape}')
    np.save('data/variables/things1854image_feas_'+modelname+'.npy',np.squeeze(image_feature_array))