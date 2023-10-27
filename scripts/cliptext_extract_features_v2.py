import sys
sys.path.append('T2I-Adapter')

from Adapter.utils import import_model_class_from_model_name_or_path
from transformers import AutoTokenizer
import numpy as np
import torch

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7] 


model_id = "stabilityai/stable-diffusion-xl-base-1.0"

#  CLIP ViT-L
tokenizer_one = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=None, use_fast=False)
text_encoder_cls_one = import_model_class_from_model_name_or_path(model_id, None)
text_encoder_one = text_encoder_cls_one.from_pretrained(model_id, subfolder="text_encoder", revision=None)

# OpenCLIP ViT-bigG
tokenizer_two = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer_2", revision=None, use_fast=False)
text_encoder_cls_two = import_model_class_from_model_name_or_path(model_id, None, subfolder="text_encoder_2")
text_encoder_two = text_encoder_cls_two.from_pretrained(model_id, subfolder="text_encoder_2", revision=None)

text_encoders = [text_encoder_one, text_encoder_two]
tokenizers = [tokenizer_one, tokenizer_two]


keys = ["train", "test"]

caps = {
    keys[0]: np.load('data/processed_data/subj{:02d}/nsd_train_cap_sub{}.npy'.format(sub,sub)), 
    keys[1]: np.load('data/processed_data/subj{:02d}/nsd_test_cap_sub{}.npy'.format(sub,sub))  
}

num_embed, num_features, num_test, num_train = 77, 2048, len(caps[keys[1]]), len(caps[keys[0]])

# concatenation of encoder outputs
clip = {
    keys[0]: np.zeros((num_train,num_embed, num_features)),
    keys[1]: np.zeros((num_test,num_embed, num_features)), 
}

# pooled text embedding from the OpenCLIP
pooled_num_embed = 1280
pooled = {
    keys[0]: np.zeros((num_train, pooled_num_embed)), 
    keys[1]: np.zeros((num_test, pooled_num_embed)), 
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():
    for data in keys:
        for i,annots in enumerate(caps[data]):
            captions = list(annots[annots!=''])
            print(i)

            prompt_embeds_list = []

            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                prompt_embeds = text_encoder(
                    text_input_ids.to(text_encoder.device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

            clip[data][i] = prompt_embeds.to('cpu').numpy().mean(0)
            pooled[data][i] = pooled_prompt_embeds.to('cpu').numpy().mean(0)
        
        np.save('data/extracted_features/subj{:02d}/nsd_cliptext_{}_v2.npy'.format(sub, data), clip[data])
        np.save('data/extracted_features/subj{:02d}/nsd_opencliptext_{}.npy'.format(sub, data), pooled[data])

    



    

