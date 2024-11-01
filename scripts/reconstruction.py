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

from samplers.test_sd15 import  center_crop, load_im_into_format_from_path, pil_to_latents
def get_jpg_paths(directory):
    jpg_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                jpg_paths.append(os.path.abspath(os.path.join(root, file)))
    return jpg_paths
def transform_image_array(image_array):
    if image_array.ndim == 2:
        # If it is a grayscale image, the channel needs to be copied
        return np.repeat(image_array[:, :, np.newaxis], 3, axis=2)
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        # If it is an RGB image, no conversion is necessary
        return image_array
    else:
        raise ValueError("The input image array format is incorrect")

def recon_test(im, sd_params = None, sd_pipe=None, type = 'ddim', store_extra = True):
    # disk to pil_image
    if isinstance(im, str): im = load_im_into_format_from_path(im)
    # pil_image to latent
    latent = pil_to_latents(pil_image= im, sd_pipe= sd_pipe)
    # latent to intermediate
    if type in ['ddim']:
        intermediate = DDIM.latent_to_intermediate(sd_pipe= sd_pipe, sd_params=sd_params, latent=latent)
    elif type in ['edict']:
        x_intermediate, y_intermediate = edict.latent_to_intermediate(sd_pipe=sd_pipe, sd_params=sd_params, latent=latent)
    elif type in ['bdia']:
        intermediate, second_intermediate = BDIA.latent_to_intermediate(sd_pipe=sd_pipe, sd_params=sd_params, latent=latent)
    elif type in ['lag','belm']:
        intermediate, second_intermediate = BELM.latent_to_intermediate(sd_pipe=sd_pipe, sd_params=sd_params, latent=latent)

    # intermediate to latent
    if type in ['ddim']:
        recon_latent = DDIM.intermediate_to_latent(sd_pipe=sd_pipe,sd_params=sd_params,intermediate=intermediate)
    elif type in ['edict']:
        if store_extra:
            recon_latent, _ = edict.intermediate_to_latent(sd_pipe=sd_pipe,sd_params=sd_params,x_intermediate=x_intermediate,y_intermediate=y_intermediate)
        else:
            recon_latent, _ = edict.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params,
                                                           x_intermediate=x_intermediate, y_intermediate=x_intermediate)
    elif type in ['bdia']:
        if store_extra:
            recon_latent = BDIA.intermediate_to_latent(sd_pipe=sd_pipe,sd_params=sd_params,intermediate = intermediate,intermediate_second= second_intermediate)
        else:
            recon_latent = BDIA.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params, intermediate=intermediate,
                                                       intermediate_second=None)
    elif type in ['lag','belm']:
        if store_extra:
            recon_latent = BELM.intermediate_to_latent(sd_pipe=sd_pipe,sd_params=sd_params,intermediate = intermediate,intermediate_second= second_intermediate)
        else:
            recon_latent = BELM.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params,
                                                                      intermediate=intermediate,
                                                                      intermediate_second=second_intermediate)


    # latent to pil_image
    pil = test_sd15.to_pil(latents= recon_latent, sd_pipe=sd_pipe)
    arr1 = transform_image_array(np.array(im))
    arr1 = arr1 / 255.0
    arr2 = np.array(pil)
    arr2 = arr2 / 255.0

    latent_arr1 = np.array(latent.to('cpu'))
    latent_arr2 = np.array(recon_latent.to('cpu'))
    mse1 = np.mean((latent_arr1 - latent_arr2) ** 2)
    print('mse between latent = ', mse1)
    mse2 = np.mean((arr1 - arr2) ** 2)
    print('mse between pil = ',mse2)
    return mse1, mse2

def ave_mse(sd_params = None, sd_pipe=None, type = 'ddim',store_extra = True, test_num=100, directory = 'xxx'):
    jpg_paths = get_jpg_paths(directory)
    mse1_list = []
    mse2_list = []
    for i,im in enumerate(jpg_paths[0:test_num]):
        print(type,'##',i)
        mse1, mse2 = recon_test(im=im,sd_params=sd_params,sd_pipe=sd_pipe,type=type,store_extra=store_extra)
        mse1_list.append(mse1)
        mse2_list.append(mse2)
    mse1_ave = sum(mse1_list) / len(mse1_list)
    mse2_ave = sum(mse2_list) / len(mse2_list)
    return mse1_ave, mse2_ave


def main():
    parser = argparse.ArgumentParser(description="sampling script for reconstruction on xxx machine.")
    parser.add_argument('--test_num', type=int, default=100)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance', type=float, default=1.0)
    parser.add_argument('--sampler_type', type = str,default='lag', choices=['lag', 'ddim', 'bdia', 'edict', 'belm'])
    parser.add_argument('--directory', type=str, default='coco2014')
    parser.add_argument('--model_id', type=str, default='xxxxx/stable-diffusion-v1-5')
    args = parser.parse_args()
    if args.sampler_type in ['bdia']:
        parser.add_argument('--bdia_gamma', type=float, default=1.0)
    if args.sampler_type in ['edict']:
        parser.add_argument('--edict_p', type=float, default=0.5)
    args = parser.parse_args()

    sampler_type = args.sampler_type
    directory = args.directory
    test_num = args.test_num
    guidance_scale = args.guidance
    num_inference_steps = args.num_inference_steps
    model_id = args.model_id

    # load model
    device = 'cuda'
    dtype = torch.float32
    sd = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    sche = DDIMScheduler(beta_end=0.012, beta_start=0.00085, beta_schedule='scaled_linear', clip_sample=False,
                         timestep_spacing='linspace', set_alpha_to_one=False)

    sd_pipe = PipelineLike(device=device, vae=sd.vae, text_encoder=sd.text_encoder, tokenizer=sd.tokenizer,
                           unet=sd.unet, scheduler=sche)
    sd_pipe.vae.to(device)
    sd_pipe.text_encoder.to(device)
    sd_pipe.unet.to(device)
    print('model loaded')

    sd_params = {'prompt': '', 'negative_prompt': '', 'seed': 1, 'guidance_scale': guidance_scale,
                 'num_inference_steps': num_inference_steps, 'width': 512, 'height': 512}
    m1, m2 = ave_mse(sd_params=sd_params,sd_pipe=sd_pipe,type=sampler_type,test_num=test_num,directory=directory)
    print('#####################  FINAL RESULT   ######################')
    print(f'reconstruction mse average across {test_num} pictures using {sampler_type} sampler:')
    print(f'{sampler_type} mse1 on latent space = ', m1)
    print(f'{sampler_type} mse2 on pixel space = ', m2)



if __name__ == '__main__':
    main()