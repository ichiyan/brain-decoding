from diffusers import StableDiffusionPipeline
from compel import Compel
import torch
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]


train_caps = np.load('data/processed_data/subj{:02d}/nsd_train_cap_sub{}.npy'.format(sub,sub)) 
test_caps = np.load('data/processed_data/subj{:02d}/nsd_test_cap_sub{}.npy'.format(sub,sub))  

num_embed, num_features, num_test, num_train = 77, 768, len(test_caps), len(train_caps)

train_clip = np.zeros((num_train,num_embed, num_features))
test_clip = np.zeros((num_test,num_embed, num_features))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_id = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, variant="fp16"
).to(device)

# pipe.enable_model_cpu_offload()

compel = Compel(
    tokenizer=pipe.tokenizer, 
    text_encoder=pipe.text_encoder, 
)

with torch.no_grad():
    for i,annots in enumerate(test_caps):
        caps = list(annots[annots!=''])
        print(i)
        prompt_embeds = compel(caps)
        test_clip[i] = prompt_embeds.to('cpu').numpy().mean(0)
    
    np.save('data/extracted_features/subj{:02d}/nsd_compel_cliptext_test.npy'.format(sub),test_clip)
        
    for i,annots in enumerate(train_caps):
        caps = list(annots[annots!=''])
        print(i)
        prompt_embeds = compel(caps)
        train_clip[i] = prompt_embeds.to('cpu').numpy().mean(0)

    np.save('data/extracted_features/subj{:02d}/nsd_compel_cliptext_train.npy'.format(sub),train_clip)


