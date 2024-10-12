import sys

import torch
import os
import json
import argparse
sys.path.append(os.getcwd())
from cn_dm.test.adjoint_state import test_sd15, lagrange_reversible, BDIA, edict, DDIM
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import glob

from cn_dm.test.adjoint_state.test_sd15 import  center_crop, load_im_into_format_from_path, pil_to_latents
def get_jpg_paths(directory):
    jpg_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                jpg_paths.append(os.path.abspath(os.path.join(root, file)))
    return jpg_paths
def transform_image_array(image_array):
    if image_array.ndim == 2:
        # 灰度图像, 需要复制通道
        return np.repeat(image_array[:, :, np.newaxis], 3, axis=2)
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        # RGB 图像, 无需转换
        return image_array
    else:
        raise ValueError("输入的图像数组格式不正确")

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
        intermediate, second_intermediate = lagrange_reversible.latent_to_intermediate(sd_pipe=sd_pipe, sd_params=sd_params, latent=latent)

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
            recon_latent = lagrange_reversible.intermediate_to_latent(sd_pipe=sd_pipe,sd_params=sd_params,intermediate = intermediate,intermediate_second= second_intermediate)
        else:
            recon_latent = lagrange_reversible.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params,
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
    # directory = "xxx/coco2014/fid_test_3W"
    jpg_paths = get_jpg_paths(directory)
    mse1_list = []
    mse2_list = []
    for i,im in enumerate(jpg_paths[0:test_num]):
        # print('##',i,'im_id = ', im)
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
    # parser.add_argument('--start_index', type=int, default=0)
    # parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    # parser.add_argument('--guidance', type=float, default=1.0)
    parser.add_argument('--sampler_type', type = str,default='lag', choices=['lag', 'ddim', 'bdia', 'edict', 'belm'])
    parser.add_argument('--save_dir', type=str, default='xxxx')
    parser.add_argument('--directory', type=str, default='xxxx')

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
    # load model
    sd_pipe, clip_model, ae_model, trans = test_sd15.load_models(torch.float32)
    print("sd-1.5 model loaded")

    sd_params = {'prompt': '', 'negative_prompt': '', 'seed': 1, 'guidance_scale': guidance_scale,
                 'num_inference_steps': num_inference_steps, 'width': 512, 'height': 512}
    m1, m2 = ave_mse(sd_params=sd_params,sd_pipe=sd_pipe,type=sampler_type,test_num=test_num,directory=directory)
    print(f'{sampler_type} mse1 on latent space = ', m1)
    print(f'{sampler_type} mse2 on pixel space = ', m2)



if __name__ == '__main__':
    main()