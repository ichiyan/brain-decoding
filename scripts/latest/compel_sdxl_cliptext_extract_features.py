import numpy as np
import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import DiffusionPipeline

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7] 



keys = ["train", "test"]

caps = {
    keys[0]: np.load('data/processed_data/subj{:02d}/nsd_train_cap_sub{}.npy'.format(sub,sub)), 
    keys[1]: np.load('data/processed_data/subj{:02d}/nsd_test_cap_sub{}.npy'.format(sub,sub))  
}

num_features1 = 768 
num_features2 = 1280
num_embed, num_features, num_test, num_train = 77, num_features1 + num_features2, len(caps[keys[1]]), len(caps[keys[0]])


# concatenation of embeddings from both encoders
clip_penultimate = {
    keys[0]: np.zeros((num_train,num_embed, num_features)),
    keys[1]: np.zeros((num_test,num_embed, num_features)), 
}

openclip_penultimate = {
    keys[0]: np.zeros((num_train, num_embed, num_features2)),
    keys[1]: np.zeros((num_test, num_embed, num_features2)), 
}

# pooled text embedding from the OpenCLIP
pooled_num_embed = 1280
pooled = {
    keys[0]: np.zeros((num_train, pooled_num_embed)), 
    keys[1]: np.zeros((num_test, pooled_num_embed)), 
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_pipe = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  variant="fp16",
  torch_dtype=torch.float16
)

refiner_pipe = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-refiner-1.0",
  variant="fp16",
  torch_dtype=torch.float16
)

base_pipe.enable_model_cpu_offload()
refiner_pipe.enable_model_cpu_offload()

base_compel = Compel(
    tokenizer=[base_pipe.tokenizer, base_pipe.tokenizer_2] ,
    text_encoder=[base_pipe.text_encoder, base_pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True]
)

refiner_compel = Compel(
    tokenizer=base_pipe.tokenizer_2,
    text_encoder=base_pipe.text_encoder_2,
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=True,
)

with torch.no_grad():
    for data in keys:
        for i,annots in enumerate(caps[data]):
            captions = list(annots[annots!=''])
            print(i)

            base_prompt_embeds, base_pooled = base_compel(captions) 
            refiner_prompt_embeds, refiner_pooled = refiner_compel(captions)

            clip_penultimate[data][i] = base_prompt_embeds.to('cpu').numpy().mean(0)
            openclip_penultimate[data][i] = refiner_prompt_embeds.to('cpu').numpy().mean(0)
            pooled[data][i] = base_pooled.to('cpu').numpy().mean(0)
        
        np.save('data/extracted_features/subj{:02d}/nsd_compel_cliptext_penultimate_concat_{}.npy'.format(sub, data), clip_penultimate[data])
        np.save('data/extracted_features/subj{:02d}/nsd_compel_opencliptext_penultimate_{}.npy'.format(sub, data), openclip_penultimate[data])
        np.save('data/extracted_features/subj{:02d}/nsd_compel_opencliptext_pooled_{}.npy'.format(sub, data), pooled[data])

    



    

