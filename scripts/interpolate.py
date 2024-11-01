import sys

import torch
import os
import json
import argparse
sys.path.append(os.getcwd())
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
import os
import random
import torch
import torchvision
from torchvision.utils import make_grid
from PIL import Image
from torchvision.transforms import ToTensor
import math

def pil_to_latent(pil):
    transform = ToTensor()
    image1 = transform(pil)
    print(image1.shape)
    image1 = image1.unsqueeze(0)
    print(image1.shape)

    image1 = image1.cpu()
    image1 = (image1 - 0.5) * 2
    image1 = image1.clamp(-1, 1)
    print(image1.shape)
    return image1

def ddim_forward(ddpm_pipe, num_inference_steps, batch_size, intermediate=None):
    dtype = torch.float32
    # torch.manual_seed(seed)
    ddpm_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')
    timesteps = ddpm_pipe.scheduler.timesteps

    xis = []
    # Sample gaussian noise to begin loop
    if isinstance(ddpm_pipe.unet.config.sample_size, int):
        image_shape = (
            batch_size,
            ddpm_pipe.unet.config.in_channels,
            ddpm_pipe.unet.config.sample_size,
            ddpm_pipe.unet.config.sample_size,
        )
    else:
        image_shape = (batch_size, ddpm_pipe.unet.config.in_channels, *ddpm_pipe.unet.config.sample_size)
    if intermediate is None:
        intermediate = torch.randn(image_shape, generator=None, device='cuda', dtype=dtype)

    xis.append(intermediate)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # print('###', i)
            noise_pred = ddpm_pipe.unet(
                intermediate,
                t,
                return_dict=False,
            )[0]

            if i < num_inference_steps - 1:
                alpha_s = ddpm_pipe.scheduler.alphas_cumprod[timesteps[i + 1]].to(torch.float32)
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[t].to(torch.float32)
            else:
                alpha_s = 1
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[t].to(torch.float32)

            sigma_s = (1 - alpha_s)**0.5
            sigma_t = (1 - alpha_t)**0.5
            alpha_s = alpha_s**0.5
            alpha_t = alpha_t**0.5

            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            intermediate = coef_xt * intermediate + coef_eps * noise_pred
            xis.append(intermediate)
    images = xis[-1]
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = ddpm_pipe.numpy_to_pil(images)
    return images


def ddim_inversion(ddpm_pipe, num_inference_steps, latent):
    dtype = torch.float32
    ddpm_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')

    xis=[]
    timesteps = ddpm_pipe.scheduler.timesteps
    xis.append(latent)
    prev_noise = None

    # print(num_inference_steps)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            index = num_inference_steps - i - 1

            time = timesteps[index + 1] if index < num_inference_steps - 1 else 1
            noise_pred = ddpm_pipe.unet(
                latent,
                time,
                return_dict=False,
            )[0]

            if index < num_inference_steps - 1:
                alpha_s = ddpm_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[timesteps[index + 1]].to(torch.float32)
            else:
                alpha_s = ddpm_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_t = 1

            sigma_s = (1 - alpha_s) ** 0.5
            sigma_t = (1 - alpha_t) ** 0.5
            alpha_s = alpha_s ** 0.5
            alpha_t = alpha_t ** 0.5

            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            latent = coef_xt * latent + coef_eps * noise_pred

            xis.append(latent)
    return xis[-1]

def belm_inversion(ddpm_pipe, num_inference_steps, latent):
    dtype = torch.float32
    ddpm_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')

    xis=[]
    timesteps = ddpm_pipe.scheduler.timesteps
    xis.append(latent)
    prev_noise = None

    # print(num_inference_steps)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            index = num_inference_steps - i - 1

            time = timesteps[index + 1] if index < num_inference_steps - 1 else 1
            noise_pred = ddpm_pipe.unet(
                latent,
                time,
                return_dict=False,
            )[0]

            if index < num_inference_steps - 1:
                alpha_i = ddpm_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_i_minus_1 = ddpm_pipe.scheduler.alphas_cumprod[timesteps[index + 1]].to(torch.float32)
            else:
                alpha_i = ddpm_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_i_minus_1 = 1

            sigma_i = (1 - alpha_i) ** 0.5
            sigma_i_minus_1 = (1 - alpha_i_minus_1) ** 0.5
            alpha_i = alpha_i ** 0.5
            alpha_i_minus_1 = alpha_i_minus_1 ** 0.5

            if i == 0:
                latent = (alpha_i / alpha_i_minus_1) * latent + (sigma_i - (alpha_i / alpha_i_minus_1) * sigma_i_minus_1)
            else:
                alpha_i_minus_2 = 1 if i == 1 else ddpm_pipe.scheduler.alphas_cumprod[timesteps[index + 2]].to(torch.float32)
                sigma_i_minus_2 = (1 - alpha_i_minus_2) ** 0.5
                alpha_i_minus_2 = alpha_i_minus_2 ** 0.5

                h_i = sigma_i / alpha_i - sigma_i_minus_1 / alpha_i_minus_1
                h_i_minus_1 = sigma_i_minus_1 / alpha_i_minus_1 - sigma_i_minus_2 / alpha_i_minus_2

                coef_x_i_minus_2 = (alpha_i / alpha_i_minus_2) * (h_i ** 2) / (h_i_minus_1 ** 2)
                coef_x_i_minus_1 = (alpha_i / alpha_i_minus_1) * (h_i_minus_1 ** 2 - h_i ** 2) / (h_i_minus_1 ** 2)
                coef_eps = alpha_i * (h_i_minus_1 + h_i) * h_i / h_i_minus_1
                latent = coef_x_i_minus_2 * xis[-2] + coef_x_i_minus_1 * xis[-1] + coef_eps * noise_pred
            xis.append(latent)
        return xis[-1], xis[-2]

def belm_forward(ddpm_pipe, num_inference_steps, batch_size, intermediate, intermediate_second):
    dtype = torch.float32
    # torch.manual_seed(seed)
    ddpm_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')
    timesteps = ddpm_pipe.scheduler.timesteps

    xis = []
    # Sample gaussian noise to begin loop
    if isinstance(ddpm_pipe.unet.config.sample_size, int):
        image_shape = (
            batch_size,
            ddpm_pipe.unet.config.in_channels,
            ddpm_pipe.unet.config.sample_size,
            ddpm_pipe.unet.config.sample_size,
        )
    else:
        image_shape = (batch_size, ddpm_pipe.unet.config.in_channels, *ddpm_pipe.unet.config.sample_size)
    if intermediate is None:
        intermediate = torch.randn(image_shape, generator=None, device='cuda', dtype=dtype)

    xis.append(intermediate)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # print('###', i)
            noise_pred = ddpm_pipe.unet(
                intermediate,
                t,
                return_dict=False,
            )[0]

            if i < num_inference_steps - 1:
                alpha_s = ddpm_pipe.scheduler.alphas_cumprod[timesteps[i + 1]].to(torch.float32)
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[t].to(torch.float32)
            else:
                alpha_s = 1
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[t].to(torch.float32)

            sigma_s = (1 - alpha_s) ** 0.5
            sigma_t = (1 - alpha_t) ** 0.5
            alpha_s = alpha_s ** 0.5
            alpha_t = alpha_t ** 0.5

            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            if i == 0:
                if intermediate_second is not None:
                    # print('have intermediate_second')
                    intermediate = intermediate_second.clone()
                else:
                    # print('dont have intermediate_second')
                    intermediate = coef_xt * intermediate + coef_eps * noise_pred
            else:
                # calculate i-1
                alpha_p = ddpm_pipe.scheduler.alphas_cumprod[timesteps[i - 1]].to(torch.float32)
                sigma_p = (1 - alpha_p) ** 0.5
                alpha_p = alpha_p ** 0.5

                # calculate t
                t_p, t_t, t_s = sigma_p / alpha_p, sigma_t / alpha_t, sigma_s / alpha_s

                # calculate delta
                delta_1 = t_t - t_p
                delta_2 = t_s - t_t
                delta_3 = t_s - t_p

                # calculate coef
                coef_1 = delta_2 * delta_3 * alpha_s / delta_1
                coef_2 = (delta_2 / delta_1) ** 2 * (alpha_s / alpha_p)
                coef_3 = (delta_1 - delta_2) * delta_3 / (delta_1 ** 2) * (alpha_s / alpha_t)

                # iterate
                intermediate = coef_1 * noise_pred + coef_2 * xis[-2] + coef_3 * xis[-1]
            xis.append(intermediate)
    images = xis[-1]
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = ddpm_pipe.numpy_to_pil(images)
    return images

def slerp(val, low, high):
    low_norm = low / low.norm(dim=(-1, -2), keepdim=True)
    high_norm = high / high.norm(dim=(-1, -2), keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(dim=(-1, -2), keepdim=True))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
    return res

def main():
    parser = argparse.ArgumentParser(description="sampling script for celeb interpolation on chongqing machine.")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--sampler_type', type = str,default='lag', choices=['lag', 'ddim', 'bdia', 'edict','belm'])
    parser.add_argument('--save_dir', type=str, default='xxx')
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--model_id', type=str, default='/xxx/ddpm-ema-celebahq-256')

    args = parser.parse_args()
    dtype = torch.float32
    save_dir = args.save_dir
    num_inference_steps = args.num_inference_steps
    test_num = args.test_num

    # load model
    model_id = args.model_id

    ddpm = DDIMPipeline.from_pretrained(model_id,torch_dtype=torch.float32)
    ddpm.unet.to('cuda')
    with torch.no_grad():
        for seed in range(test_num):
            print('seed=', seed)
            if isinstance(ddpm.unet.config.sample_size, int):
                image_shape = (
                    1,
                    ddpm.unet.config.in_channels,
                    ddpm.unet.config.sample_size,
                    ddpm.unet.config.sample_size,
                )
            else:
                image_shape = (1, ddpm.unet.config.in_channels, *ddpm.unet.config.sample_size)
            intermediate1_1 = torch.randn(image_shape, generator=None, device='cuda', dtype=dtype).to('cuda')
            intermediate2_1 = torch.randn(image_shape, generator=None, device='cuda', dtype=dtype).to('cuda')

            for num_inference_steps in [num_inference_steps]:
                print(f'belm{num_inference_steps}')
                grids = []
                for percen in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    print('Interpolation coefficient:', percen)
                    intermediatep = slerp(percen, intermediate1_1, intermediate2_1)

                    images = belm_forward(ddpm_pipe=ddpm, num_inference_steps=num_inference_steps, batch_size=1,
                                          intermediate=intermediatep, intermediate_second=None)
                    imp = images[0]
                    grids.append(imp)

                grids = [ToTensor()(im) for im in grids]
                images_tensor = torch.stack(grids)
                grid = make_grid(images_tensor, nrow=11)
                to_pil = torchvision.transforms.ToPILImage()
                img = to_pil(grid)
                img.save(os.path.join(save_dir,f'seed{seed}_belm_inter__{num_inference_steps}.png'))




if __name__ == '__main__':
    main()

