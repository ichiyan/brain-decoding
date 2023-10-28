import sys
sys.path.append('latent_vae')


import torch
from pipeline import Pipeline
from omegaconf import OmegaConf
from ml_collections import ConfigDict 
import argparse
from PIL import Image
import numpy as np


from vae.config.imagenet256_kl8 import get_config as imagenet_config

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-bs", "--bs",help="Batch Size",default=1)
parser.add_argument("--dae_bs", help='batch size for DAE decoding', type=int, default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]
batch_size=int(args.bs)
dae_batchsize=int(args.dae_bs)


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


alpha = 40

pred_latents = np.load('data/predicted_features/subj{:02d}/nsd_lvae_nsdgeneral_pred_sub{}_alpha{}k.npy'.format(sub,sub,alpha))
pred_latents = torch.from_numpy(pred_latents).unflatten(dim=1, sizes=(4,32,32)).float()
print(pred_latents.shape)
output_ims = []


# currently working only for bs = 1
for ndx in range(int(np.ceil(len(pred_latents)/batch_size))):
    print(ndx*batch_size)

    with torch.no_grad():
        for i in range(batch_size // dae_batchsize):
            lats = pred_latents[ndx].unsqueeze(0)
            lats = lats[i*dae_batchsize : i*dae_batchsize+dae_batchsize]
            ims = pipe.dae.decode(lats.to(device))
            ims = ims.float().cpu().permute(0, 2, 3, 1)
            output_ims.append(ims)

output_ims = np.concatenate(tuple(output_ims), axis=0)
output_ims = output_ims * 0.5 + 0.5
output_ims = np.clip(output_ims, 0., 1.) * 255
output_ims = output_ims.astype('uint8')

print("Saving images...")
for ndx, img in enumerate(output_ims):
    print(ndx)
    im = Image.fromarray(img)
    im = im.resize((512,512),resample=3)
    im.save('results/vdvae/subj{:02d}/lvae_stim_{}k/{}.png'.format(sub,alpha,ndx))







