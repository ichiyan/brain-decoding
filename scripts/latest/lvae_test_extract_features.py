import sys
sys.path.append('latent_vae')

import torchvision
import torch
from pipeline import Pipeline
from omegaconf import OmegaConf
from ml_collections import ConfigDict 
import argparse
from PIL import Image
from tensorflow.io import gfile
import time
import os
import warnings
import numpy as np


from vae.config.bedroom import get_config as bedroom_config
from vae.config.church import get_config as church_config
from vae.config.imagenet256_kl8 import get_config as imagenet_config

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-bs", "--bs",help="Batch Size",default=30)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]
batch_size=int(args.bs)


# latent vae model
INFO = {
    "imagenet": ConfigDict({
        "dae_config": "./latent_vae/dae/config/imagenet.yaml",
        "vae_config": imagenet_config(),
        "weights_name": "./latent_vae/models/inet_pipeline.pt",
        "weights_id": "1-Td-danBSRX4IhlXAD_CtgFARbu4C9hH",
    }),
}

model_name = "imagenet"
config = INFO[model_name]

vae_config = config.vae_config
dae_config = OmegaConf.load(config.dae_config)["model"]
pipe = Pipeline(vae_config, dae_config)
if torch.cuda.is_available():
    print('cuda')
    pipe = pipe.to('cuda')

pipe.load_from_ckpt(config.weights_name) 

if pipe.vae.num_classes:
    device = next(pipe.parameters()).device


class batch_generator_external_images(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)


    def __getitem__(self,idx):
        img = Image.fromarray(self.im[idx])
        img = transforms.functional.resize(img,(256,256))
        img = torch.tensor(np.array(img)).float().permute(2,0,1)
        img = img / 255.0 
        img = 2 * (img - 0.5)
        return img

    def __len__(self):
        return  len(self.im)
    

# stimulus images
image_path = 'data/processed_data/subj{:02d}/nsd_train_stim_sub{}.npy'.format(sub,sub)
train_images = batch_generator_external_images(data_path = image_path)

image_path = 'data/processed_data/subj{:02d}/nsd_test_stim_sub{}.npy'.format(sub,sub)
test_images = batch_generator_external_images(data_path = image_path)

# trainloader = DataLoader(train_images,batch_size,shuffle=False)
# testloader = DataLoader(test_images,batch_size,shuffle=False)

keys = ["test", "train"]

dataloaders = {
    keys[0]: DataLoader(test_images,batch_size,shuffle=False),
    keys[1]: DataLoader(train_images,batch_size,shuffle=False),
}


latents = {
    keys[0]: [],
    keys[1]: [],
}

# test_latents = []

for data in keys:
    print(data)
    for i,img in enumerate(dataloaders[data]):
        print(i*batch_size)
        latent_dist = pipe.dae.encode(img.to(device))
        latent_samples = latent_dist.sample()
        latent_samples *= 0.18215 

        label = None

        if label is None:
            label = torch.randint(size=[batch_size], low=0, high=pipe.vae.num_classes).to(device)
        else:
            label = (torch.ones([batch_size]) * int(label)).int().to(device)  

        uncond_label = torch.full_like(label, pipe.vae.num_classes).to(device)
        mask = torch.greater(torch.rand(label.shape), 0.9).int().to(device)
        cf_guidance_label = label*(1-mask) + mask*uncond_label

        label = pipe.vae.embed(label).to(device)
        cf_guidance_label = pipe.vae.embed(cf_guidance_label).to(device)
        generator_label = (label, cf_guidance_label)
        x = torch.tile(pipe.vae.initial_x, [img.shape[0], 1, 1, 1]).to(device)

        acts = pipe.vae.encoder.forward(latent_samples, label)
        KLs = []
        SR_Losses = 0.

        with torch.no_grad():
            for i, decoderlevel in enumerate(pipe.vae.layer_list):
                j = -i+len(pipe.vae.nlayers)-1
                x, kls, sr_losses = decoderlevel(x, acts[j], generator_label)
                KLs.extend(kls)
                SR_Losses += sr_losses

            x = pipe.vae.outproj(x)
            latents = x[:, :pipe.vae.datadim, ...]
            latents = (latents * pipe.vae_config.dataset.scale) + pipe.vae_config.dataset.shift

        latents[data].append(latents)

latents[keys[0]] = np.concatenate(latents[keys[0]] )
latents[keys[1]] = np.concatenate(latents[keys[1]] )

np.savez("data/extracted_features/subj{:02d}/nsd_lvae_features.npz".format(sub),train_latents=latents[keys[1]],test_latents=latents[keys[0]])


