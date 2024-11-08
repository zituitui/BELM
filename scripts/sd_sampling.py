import sys

import torch
import os
import json
import argparse
sys.path.append(os.getcwd())
from samplers import test_sd15, BELM, BDIA, edict, DDIM
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import glob
from diffusers import StableDiffusionPipeline, DDIMScheduler
from samplers.test_sd15 import  center_crop, load_im_into_format_from_path, pil_to_latents
from samplers.utils import PipelineLike

def main():
    parser = argparse.ArgumentParser(description="sampling script for COCO14 on chongqing machine.")
    parser.add_argument('--num_inference_steps', type=int, default=200)
    parser.add_argument('--guidance', type=float, default=4.0)
    parser.add_argument('--sampler_type', type = str,default='lag', choices=['lag', 'ddim', 'bdia', 'edict', 'belm'])
    parser.add_argument('--save_dir', type=str, default='xx')
    parser.add_argument('--model_id', type=str, default='xxxxx/stable-diffusion-v1-5')
    parser.add_argument('--prompt', type=str, default='A dog')
    parser.add_argument('--negative_prompt', type=str, default='')
    parser.add_argument('--bdia_gamma', type=float, default=0.96)
    parser.add_argument('--edict_p', type=float, default=0.93)
    args = parser.parse_args()

    sampler_type = args.sampler_type
    guidance_scale = args.guidance
    num_inference_steps = args.num_inference_steps
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    model_id = args.model_id
    device = 'cuda'
    dtype = torch.float32

    # load model
    sd = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    sche = DDIMScheduler(beta_end=0.012, beta_start=0.00085, beta_schedule='scaled_linear', clip_sample=False,
                         timestep_spacing='linspace', set_alpha_to_one=False)

    sd_pipe = PipelineLike(device=device, vae=sd.vae, text_encoder=sd.text_encoder, tokenizer=sd.tokenizer,
                           unet=sd.unet, scheduler=sche)
    sd_pipe.vae.to(device)
    sd_pipe.text_encoder.to(device)
    sd_pipe.unet.to(device)
    print('model loaded')

    # intermediate to latent
    sd_params = {'prompt': prompt, 'negative_prompt': negative_prompt, 'seed': 38018716,
                 'guidance_scale': guidance_scale,
                 'num_inference_steps': num_inference_steps, 'width': 512, 'height': 512}
    if sampler_type in ['ddim']:
        result_latent = DDIM.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params, intermediate=None,freeze_step=0)
    elif sampler_type in ['edict']:
        result_latent, _ = edict.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params,x_intermediate=None,y_intermediate=None,p=args.edict_p,freeze_step=0)
    elif sampler_type in ['bdia']:
        result_latent = BDIA.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params, intermediate=None,
                             intermediate_second=None,gamma= args.bdia_gamma,freeze_step=0)
    elif sampler_type in ['lag', 'belm']:
        result_latent = BELM.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params,
                                                                      intermediate=None,
                                                                     intermediate_second=None,freeze_step=0)
    pil = test_sd15.to_pil(latents=result_latent, sd_pipe=sd_pipe)
    pil.save(os.path.join(args.save_dir,
        f'{sampler_type}_infer{num_inference_steps}_g{guidance_scale}.png'))
    print('sampling finished')


if __name__ == '__main__':
    main()