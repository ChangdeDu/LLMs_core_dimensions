# import vissl
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os

# from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict

from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.models import build_model
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights

# Config is located at vissl/configs/config/pretrain/simclr/simclr_8node_resnet.yaml.
# All other options override the simclr_8node_resnet.yaml config.

cfg = [
  'config=pretrain/simclr/simclr_8node_resnet.yaml',
  'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/nfs/diskstation/DataStation/ChangdeDu/resnet_simclr.torch', # Specify path for the model weights.
  'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
  'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk.
  'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
  'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
  'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["res5avg", ["Identity", []]]]' # Extract only the res5avg features.
]

# Compose the hydra configuration.
cfg = compose_hydra_configuration(cfg)
# Convert to AttrDict. This method will also infer certain config options
# and validate the config is valid.
_, cfg = convert_to_attrdict(cfg)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model_sets= ['simclr']
for modelname in model_sets:
    model = build_model(cfg.MODEL, cfg.OPTIMIZER)
    # Load the checkpoint weights.
    weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)

    # Initializei the model with the simclr model weights.
    init_model_from_consolidated_weights(
        config=cfg,
        model=model,
        state_dict=weights,
        state_dict_key_name="classy_state_dict",
        skip_layers=[],  # Use this if you do not want to load all layers
    )

    print("Weights have loaded")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    features = []
    for i in range(1854):
        image_path = f"data/THINGS_visual_stimuli_1854/image_{i+1}_ori.jpg"
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model(image_tensor)
            feature = feature[0]
        feature_flatten = feature.cpu().flatten().numpy()
        features.append(feature_flatten)
        print(f'Image: {i + 1}, Pooled shape: {feature_flatten.shape}')
        image_feature_array = np.array(features)
        print(f'feature_array shape: {image_feature_array.shape}')
    np.save('data/variables/things1854image_feas_'+modelname+'.npy',np.squeeze(image_feature_array))