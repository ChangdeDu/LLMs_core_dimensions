import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as L
from torchvision import transforms
from src.gabor_feature_extractor import Gaborizer
from src.torch_fwrf import get_value, set_value
from src.rf_grid import linspace, logspace
import src.numpy_utility as pnu

print ('#device:', torch.cuda.device_count())
print ('device#:', torch.cuda.current_device())
print ('device name:', torch.cuda.get_device_name(torch.cuda.current_device()))

device = torch.device("cuda:0") #cuda
torch.backends.cudnn.enabled=True

print ('\ntorch:', torch.__version__)
print ('cuda: ', torch.version.cuda)
print ('cudnn:', torch.backends.cudnn.version())
print ('dtype:', torch.get_default_dtype())

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

class add_nonlinearity(L.Module):
    def __init__(self, _fmaps_fn, _nonlinearity):
        super(add_nonlinearity, self).__init__()
        self.fmaps_fn = _fmaps_fn
        self.nl_fn = _nonlinearity
    def forward(self, _x):
        return [self.nl_fn(_fm) for _fm in self.fmaps_fn(_x)]

n_ori = 6
cyc_per_stim = logspace(12)(3., 72.) # 6-92
_gaborizer = Gaborizer(num_orientations=n_ori, cycles_per_stim=cyc_per_stim,
          pix_per_cycle=4.13, cycles_per_radius=.7,
          radii_per_filter=4, complex_cell=True, pad_type='half',
          crop=False)#.to(device)

_fmaps_fn = add_nonlinearity(_gaborizer, lambda x: torch.log(1+torch.sqrt(x)))
plt.figure(figsize=(8, 4))
for k,_p in enumerate(_fmaps_fn.parameters()):
    plt.subplot(1,2,k+1)
    plt.imshow(pnu.mosaic_vis(get_value(_p)[:,0], pad=1), interpolation='None', cmap='jet')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)


features = []
for i in range(1854):
    image_path = f"data/THINGS_visual_stimuli_1854/image_{i + 1}_ori.jpg"
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)
    image_tensor_np = image_tensor.numpy()
    _x = torch.tensor(image_tensor_np)#.to(device)  # the input variable.
    _fmaps = _gaborizer(_x)
    feature = _fmaps[0]
    print(feature.size())
    feature_flatten = feature.flatten().detach().numpy()
    features.append(feature_flatten)
    print(f'Image: {i + 1}, Pooled shape: {feature_flatten.shape}')
    image_feature_array = np.array(features)
    print(f'feature_array shape: {image_feature_array.shape}')
np.save('data/variables/things1854image_feas_gabor.npy',np.squeeze(image_feature_array))

plt.figure(figsize=(36,12))
for k,_fm in enumerate(_fmaps[:12]):
    plt.subplot(2,6,k+1)
    plt.imshow(pnu.mosaic_vis(get_value(_fm)[0], pad=1), interpolation='None', cmap='jet')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

