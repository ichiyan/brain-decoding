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
parser.add_argument("-bs", "--bs",help="Batch Size",default=30)
parser.add_argument("--dae_bs", help='batch size for DAE decoding', type=int, default=2)
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


pred_latents = np.load('data/predicted_features/subj{:02d}/nsd_lvae_nsdgeneral_pred_sub{}_alpha100k.npy'.format(sub,sub))
output_ims = []

for ndx in range(int(np.ceil(len(pred_latents)/batch_size))):
    print(i*batch_size)

    with torch.no_grad():
        for i in range(batch_size // dae_batchsize):
            lats = pred_latents[ndx][i*dae_batchsize : i*dae_batchsize+dae_batchsize]
            ims = pipe.dae.decode(lats)
            ims = ims.float().cpu().permute(0, 2, 3, 1)
            output_ims.append(ims)

output_ims = np.concatenate(tuple(output_ims), axis=0)
output_ims = output_ims * 0.5 + 0.5
output_ims = np.clip(output_ims, 0., 1.) * 255
output_ims = output_ims.astype('uint8')

for ndx, img in enumerate(output_ims):
    im = Image.fromarray(img)
    im = im.resize((512,512),resample=3)
    im.save('results/vdvae/subj{:02d}/lvae_stim/{}.png'.format(sub,ndx))



#   samp = sample_from_hier_latents(input_latent,range(i*batch_size,(i+1)*batch_size))
#   px_z = ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
#   sample_from_latent = ema_vae.decoder.out_net.sample(px_z)
#   upsampled_images = []
#   for j in range(len(sample_from_latent)):
#       im = sample_from_latent[j]
#       im = Image.fromarray(im)
#       im = im.resize((512,512),resample=3)
#       im.save('results/vdvae/subj{:02d}/stim_{}l/{}.png'.format(sub,num_latents,i*batch_size+j))



