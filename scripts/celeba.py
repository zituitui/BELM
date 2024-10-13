import sys

import torch
import os
import json
import argparse
sys.path.append(os.getcwd())
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

def ddim_forward(ddpm_pipe, seed, num_inference_steps, states=None):

    dtype = torch.float32
    # torch.manual_seed(seed)
    ddpm_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')
    timesteps = ddpm_pipe.scheduler.timesteps

    xis = []
    # Sample gaussian noise to begin loop
    if isinstance(ddpm_pipe.unet.config.sample_size, int):
        image_shape = (
            1,
            ddpm_pipe.unet.config.in_channels,
            ddpm_pipe.unet.config.sample_size,
            ddpm_pipe.unet.config.sample_size,
        )
    else:
        image_shape = (1, ddpm_pipe.unet.config.in_channels, *ddpm_pipe.unet.config.sample_size)
    states = torch.randn(image_shape, generator=None, device='cuda', dtype=dtype)

    xis.append(states)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # print('###', i)
            noise_pred = ddpm_pipe.unet(
                states,
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
            states = coef_xt * states + coef_eps * noise_pred
            xis.append(states)
    image = xis[-1]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = ddpm_pipe.numpy_to_pil(image)
    return image

def belm_forward(ddpm_pipe, batch_size, num_inference_steps, states=None):
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
    states = torch.randn(image_shape, generator=None, device='cuda', dtype=dtype)

    xis.append(states)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # print('###', i)
            noise_pred = ddpm_pipe.unet(
                states,
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
            if i == 0:
                states = coef_xt * states + coef_eps * noise_pred
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
                states = coef_1 * noise_pred + coef_2 * xis[-2] + coef_3 * xis[-1]

            xis.append(states)
    image = xis[-1]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = ddpm_pipe.numpy_to_pil(image)
    return image

def bdia_forward(ddpm_pipe, batch_size, num_inference_steps, states=None, gamma = 1.0):
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
    states = torch.randn(image_shape, generator=None, device='cuda', dtype=dtype)

    xis.append(states)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # print('###', i)
            noise_pred = ddpm_pipe.unet(
                states,
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
            if i == 0:
                states = coef_xt * states + coef_eps * noise_pred
            else:
                alpha_p = ddpm_pipe.scheduler.alphas_cumprod[timesteps[i - 1]].to(torch.float32)
                sigma_p = (1 - alpha_p) ** 0.5
                alpha_p = alpha_p ** 0.5
                coef_xt = coef_xt - gamma * alpha_p / alpha_t
                coef_eps_2 = sigma_p - sigma_t * alpha_p / alpha_t
                coef_eps = coef_eps - gamma * coef_eps_2
                states = gamma * xis[-2] + coef_xt * xis[-1] + coef_eps * noise_pred

            xis.append(states)
    image = xis[-1]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = ddpm_pipe.numpy_to_pil(image)
    return image

def edict_forward(ddpm_pipe, batch_size, num_inference_steps, states=None, p = 0.93):
    dtype = torch.float32
    # torch.manual_seed(seed)
    ddpm_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')
    timesteps = ddpm_pipe.scheduler.timesteps

    xis = []
    yis = []
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
    x_states = torch.randn(image_shape, generator=None, device='cuda', dtype=dtype)
    y_states = x_states.clone()
    xis.append(x_states)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # print('###', i)
            noise_pred = ddpm_pipe.unet(
                x_states,
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
            x_inter = coef_xt * x_states + coef_eps * noise_pred
            noise_pred = ddpm_pipe.unet(
                x_inter,
                t,
                return_dict=False,
            )[0]
            y_inter = coef_xt * y_states + coef_eps * noise_pred
            x_states = p * x_inter + (1.0 - p) * y_inter
            y_states = p * y_inter + (1.0 - p) * x_states

            xis.append(x_states)
            yis.append(y_states)

    image = xis[-1]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = ddpm_pipe.numpy_to_pil(image)
    return image

def main():
    parser = argparse.ArgumentParser(description="sampling script for COCO14 on chongqing machine.")
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--sampler_type', type = str,default='lag', choices=['lag', 'ddim', 'bdia', 'edict','belm'])
    parser.add_argument('--save_dir', type=str, default='xxxx')
    parser.add_argument('--model_id', type=str, default='xxxxx/ddpm-ema-celebahq-256')
    parser.add_argument('--bdia_gamma', type=float, default=0.5)
    parser.add_argument('--edict_p', type=float, default=0.5)
    args = parser.parse_args()

    start_index = args.start_index
    batch_size = args.batch_size
    sampler_type = args.sampler_type
    test_num = args.test_num
    num_inference_steps = args.num_inference_steps
    gamma = args.bdia_gamma
    p = args.edict_p
    model_id = args.model_id

    ddpm = DDIMPipeline.from_pretrained(model_id,torch_dtype=torch.float32)
    ddpm.unet.to('cuda')
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with torch.no_grad():
        for seed in range(start_index,start_index+test_num):
            print('prepare to sample')
            if sampler_type in ['lag','belm']:
                images = belm_forward(ddpm_pipe=ddpm,batch_size=batch_size,num_inference_steps=num_inference_steps)
                for i,image in enumerate(images):
                    image.save(os.path.join(save_dir, f"belm_celebhq_inference{num_inference_steps}_seed{seed}_{i}.png"))
                print(f"belm batch##{seed},done")
            elif sampler_type in ['ddim']:
                images = ddpm(num_inference_steps = num_inference_steps, batch_size = batch_size).images
                for i,image in enumerate(images):
                    image.save(os.path.join(save_dir, f"ddim_celebhq_inference_pipe{num_inference_steps}_seed{seed}_{i}.png"))
                print(f"ddim batch##{seed},done")
            elif sampler_type in ['bdia']:
                images = bdia_forward(ddpm_pipe=ddpm,batch_size=batch_size,num_inference_steps=num_inference_steps,gamma=gamma)
                for i,image in enumerate(images):
                    image.save(os.path.join(save_dir, f"bdia_celebhq_inference_pipe{num_inference_steps}_seed{seed}_{i}.png"))
                print(f"bdia##{seed},done")
            elif sampler_type in ['edict']:
                print(f"edict##{seed},ready")
                images = edict_forward(ddpm_pipe=ddpm, batch_size=batch_size, num_inference_steps=num_inference_steps, p=p)
                for i, image in enumerate(images):
                    image.save(
                        os.path.join(save_dir, f"edict_celebhq_inference_pipe{num_inference_steps}_seed{seed}_{i}.png"))
                print(f"edict##{seed},done")

if __name__ == '__main__':
    main()
