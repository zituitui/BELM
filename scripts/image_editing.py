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
    parser.add_argument('--num_inference_steps', type=int, default=100)
    parser.add_argument('--freeze_step', type=int, default=30)
    parser.add_argument('--guidance', type=float, default=4.5)
    parser.add_argument('--sampler_type', type = str,default='lag', choices=['lag', 'ddim', 'bdia', 'edict', 'belm'])
    parser.add_argument('--save_dir', type=str, default='xx')
    parser.add_argument('--model_id', type=str, default='xxxxx/stable-diffusion-v1-5')
    parser.add_argument('--ori_im_path', type=str, default='images/imagenet_dog_1.jpg')
    parser.add_argument('--ori_prompt', type=str, default='A dog')
    parser.add_argument('--res_prompt', type=str, default='A Dalmatian')
    parser.add_argument('--bdia_gamma', type=float, default=0.96)
    parser.add_argument('--edict_p', type=float, default=0.93)
    args = parser.parse_args()

    freeze_step = args.freeze_step
    sampler_type = args.sampler_type
    guidance_scale = args.guidance
    num_inference_steps = args.num_inference_steps
    ori_prompt = args.ori_prompt
    res_prompt = args.res_prompt
    ori_im_path = args.ori_im_path
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

    negative_prompt = ''
    sd_params = {'prompt':ori_prompt, 'negative_prompt':negative_prompt, 'seed':38018716, 'guidance_scale':guidance_scale, 'num_inference_steps':num_inference_steps , 'width':512, 'height':512}

    # disk to pil_image
    im = ori_im_path
    if isinstance(im, str): im = load_im_into_format_from_path(im)

    # pil_image to latent
    latent = pil_to_latents(pil_image=im, sd_pipe=sd_pipe)

    # latent to intermediate
    if sampler_type in ['ddim']:
        intermediate = DDIM.latent_to_intermediate(sd_pipe=sd_pipe, sd_params=sd_params, latent=latent,freeze_step=freeze_step)
    elif sampler_type in ['edict']:
        x_intermediate, y_intermediate = edict.latent_to_intermediate(sd_pipe=sd_pipe, sd_params=sd_params,
                                                                      latent=latent,freeze_step=freeze_step,p=args.edict_p)
    elif sampler_type in ['bdia']:
        intermediate, second_intermediate = BDIA.latent_to_intermediate(sd_pipe=sd_pipe, sd_params=sd_params,
                                                                        latent=latent,gamma= args.bdia_gamma ,freeze_step=freeze_step)
    elif sampler_type in ['lag', 'belm']:
        intermediate, second_intermediate = BELM.latent_to_intermediate(sd_pipe=sd_pipe,
                                                                                       sd_params=sd_params,
                                                                                       latent=latent,
                                                                                       freeze_step=freeze_step)

    # intermediate to latent
    sd_params = {'prompt': res_prompt, 'negative_prompt': negative_prompt, 'seed': 38018716,
                 'guidance_scale': guidance_scale,
                 'num_inference_steps': num_inference_steps, 'width': 512, 'height': 512}
    if sampler_type in ['ddim']:
        recon_latent = DDIM.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params, intermediate=intermediate,freeze_step=freeze_step)
    elif sampler_type in ['edict']:
        recon_latent, _ = edict.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params,x_intermediate=x_intermediate,y_intermediate=y_intermediate,p=args.edict_p,freeze_step=freeze_step)
    elif sampler_type in ['bdia']:
        recon_latent = BDIA.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params, intermediate=intermediate,
                             intermediate_second=second_intermediate,gamma= args.bdia_gamma,freeze_step=freeze_step)
    elif sampler_type in ['lag', 'belm']:
        recon_latent = BELM.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params,
                                                                      intermediate=intermediate,
                                                                     intermediate_second=second_intermediate,freeze_step=freeze_step)
    pil = test_sd15.to_pil(latents=recon_latent, sd_pipe=sd_pipe)
    pil.save(os.path.join(args.save_dir,
        f'ori_{ori_prompt}_res_{res_prompt}_{sampler_type}_infer{num_inference_steps}_free{freeze_step}_g{guidance_scale}.png'))



if __name__ == '__main__':
    main()