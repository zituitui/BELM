import torch
from cn_dm.test.adjoint_state.test_sd15 import to_pil

def intermediate_to_latent(sd_pipe, sd_params, intermediate=None, freeze_step = 0):
    prompt = sd_params['prompt']
    negative_prompt = sd_params['negative_prompt']
    seed = sd_params['seed']
    guidance_scale = sd_params['guidance_scale']
    num_inference_steps = sd_params['num_inference_steps']
    width = sd_params['width']
    height = sd_params['height']
    dtype = torch.float32

    prompt_embeds = sd_pipe._encode_prompt(
        prompt,
        'cuda',
        guidance_scale > 1.0,
        negative_prompt
    )

    torch.manual_seed(seed)
    sd_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')
    timesteps = sd_pipe.scheduler.timesteps

    xis = []
    do_classifier_free_guidance = guidance_scale > 1.0
    if intermediate is None:
        print('intermediate are None')
        raise Exception('intermediate none')

    xis.append(intermediate)
    prev_noise = None
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            if i < freeze_step:
                continue
            # print('i = ',i,'t = ',t)
            # print(sd_pipe.scheduler.alphas_cumprod[timesteps[i]].to(torch.float32))
            latent_model_input = torch.cat([intermediate] * 2) if do_classifier_free_guidance else intermediate
            noise_pred = sd_pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if i < num_inference_steps - 1:
                alpha_s = sd_pipe.scheduler.alphas_cumprod[timesteps[i + 1]].to(torch.float32)
                alpha_t = sd_pipe.scheduler.alphas_cumprod[t].to(torch.float32)
            else:
                alpha_s = 1
                alpha_t = sd_pipe.scheduler.alphas_cumprod[t].to(torch.float32)

            sigma_s = (1 - alpha_s)**0.5
            sigma_t = (1 - alpha_t)**0.5
            alpha_s = alpha_s**0.5
            alpha_t = alpha_t**0.5

            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            intermediate = coef_xt * intermediate + coef_eps * noise_pred

            xis.append(intermediate)
    return xis[-1]

def latent_to_intermediate(sd_pipe, sd_params, latent = None, freeze_step = 0):
    prompt = sd_params['prompt']
    negative_prompt = sd_params['negative_prompt']
    seed = sd_params['seed']
    guidance_scale = sd_params['guidance_scale']
    num_inference_steps = sd_params['num_inference_steps']
    width = sd_params['width']
    height = sd_params['height']
    dtype = torch.float32

    prompt_embeds = sd_pipe._encode_prompt(
        prompt,
        'cuda',
        guidance_scale > 1.0,
        negative_prompt
    )

    torch.manual_seed(seed)
    sd_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')
    timesteps = sd_pipe.scheduler.timesteps

    xis = []
    do_classifier_free_guidance = guidance_scale > 1.0
    if latent is None:
        shape = (1, 4, 64, 64)
        latent = torch.randn(shape, generator=None, device='cuda', dtype=dtype)
        print('intermediate are None')

    xis.append(latent)
    prev_noise = None
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            if i >= num_inference_steps - freeze_step:
                continue
            index = num_inference_steps - i - 1
            # print('i = ', i, 't = ', t,'index = ',index)
            # print(sd_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32))
            latent_model_input = torch.cat([latent] * 2) if do_classifier_free_guidance else latent
            time = timesteps[index+1] if index < num_inference_steps - 1 else 1
            noise_pred = sd_pipe.unet(
                latent_model_input,
                time,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if index < num_inference_steps - 1:
                alpha_s = sd_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_t = sd_pipe.scheduler.alphas_cumprod[timesteps[index+1]].to(torch.float32)
            else:
                alpha_s = sd_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_t = 1

            sigma_s = (1 - alpha_s)**0.5
            sigma_t = (1 - alpha_t)**0.5
            alpha_s = alpha_s**0.5
            alpha_t = alpha_t**0.5

            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            latent = coef_xt * latent + coef_eps * noise_pred

            xis.append(latent)
    return xis[-1]