import torch
from samplers.test_sd15 import to_pil

def rev_forward(sd_pipe, sd_params, latents=None, n_mid = 0, gamma = 0.5 ):

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
    if latents is None:
        shape = (1, 4, 64, 64)
        latents = torch.randn(shape, generator=None, device='cuda', dtype=dtype)
        print('latents are None')

    xis.append(latents)
    prev_noise = None
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # print('###', i)
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            noise_pred = sd_pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # if not prev_noise is None:
            #     print(i, abs(prev_noise - noise_pred).mean())
            #     new_noise_pred = 0.5 * noise_pred + 0.5 * prev_noise
            # else:
            #     new_noise_pred = noise_pred
            # prev_noise = noise_pred.clone()
            # new_noise_pred = noise_pred

            if i < num_inference_steps - 1:
                alpha_s = sd_pipe.scheduler.alphas_cumprod[timesteps[i + 1]].to(torch.float32)
                alpha_t = sd_pipe.scheduler.alphas_cumprod[t].to(torch.float32)
            else:
                alpha_s = 1
                alpha_t = sd_pipe.scheduler.alphas_cumprod[t].to(torch.float32)

            # n_mid = 10
            # if i > n_mid:
            #     alpha_t = sd_pipe.scheduler.alphas_cumprod[timesteps[i-1]].to(torch.float32)
            sigma_s = (1 - alpha_s)**0.5
            sigma_t = (1 - alpha_t)**0.5
            alpha_s = alpha_s**0.5
            alpha_t = alpha_t**0.5

            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            if i == 0 or i <= n_mid:
                latents = coef_xt * latents + coef_eps * noise_pred
            else:
                alpha_p = sd_pipe.scheduler.alphas_cumprod[timesteps[i - 1]].to(torch.float32)
                sigma_p = (1 - alpha_p) ** 0.5
                alpha_p = alpha_p ** 0.5
                coef_xt = coef_xt - gamma * alpha_p / alpha_t
                coef_eps_2 = sigma_p - sigma_t * alpha_p / alpha_t
                coef_eps = coef_eps - gamma * coef_eps_2
                # print(latents.shape)
                # print(xis[-1].shape)
                assert torch.equal(latents, xis[-1])
                latents = gamma * xis[-2] + coef_xt * xis[-1] + coef_eps * noise_pred
            xis.append(latents)
            prev_noise = noise_pred.clone()
    return to_pil(xis[-1], sd_pipe)

def intermediate_to_latent(sd_pipe, sd_params, intermediate=None, intermediate_second = None, gamma = 0.5, freeze_step = 0):

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
        shape = (1, 4, 64, 64)
        intermediate = torch.randn(shape, generator=None, device='cuda', dtype=dtype)
        print('latents are None')

    xis.append(intermediate)
    prev_noise = None
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            if i < freeze_step:
                continue
            # print('###', i)
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
            if i == freeze_step:
                if intermediate_second is not None:
                    print('have intermediate_second')
                    intermediate = intermediate_second.clone()
                else:
                    print('dont have intermediate_second')
                    intermediate = coef_xt * intermediate + coef_eps * noise_pred
            else:
                alpha_p = sd_pipe.scheduler.alphas_cumprod[timesteps[i - 1]].to(torch.float32)
                sigma_p = (1 - alpha_p) ** 0.5
                alpha_p = alpha_p ** 0.5
                coef_xt = coef_xt - gamma * alpha_p / alpha_t
                coef_eps_2 = sigma_p - sigma_t * alpha_p / alpha_t
                coef_eps = coef_eps - gamma * coef_eps_2
                # print(latents.shape)
                # print(xis[-1].shape)
                # assert torch.equal(intermediate, xis[-1])
                intermediate = gamma * xis[-2] + coef_xt * xis[-1] + coef_eps * noise_pred
            xis.append(intermediate)
    return xis[-1]

def latent_to_intermediate(sd_pipe, sd_params, latent=None, gamma = 0.5, freeze_step = 0):

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
        print('latents are None')

    xis.append(latent)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            if i >= num_inference_steps - freeze_step:
                continue
            # print('###', i)
            index = num_inference_steps - i - 1
            time = timesteps[index + 1] if index < num_inference_steps - 1 else 1
            latent_model_input = torch.cat([latent] * 2) if do_classifier_free_guidance else latent
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
                alpha_i = sd_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_i_minus_1 = sd_pipe.scheduler.alphas_cumprod[timesteps[index + 1]].to(torch.float32)
            else:
                alpha_i = sd_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_i_minus_1 = 1

            sigma_i = (1 - alpha_i)**0.5
            sigma_i_minus_1 = (1 - alpha_i_minus_1)**0.5
            alpha_i = alpha_i**0.5
            alpha_i_minus_1 = alpha_i_minus_1**0.5


            if i == 0:
                latent = (alpha_i/alpha_i_minus_1)*latent+(sigma_i-(alpha_i/alpha_i_minus_1)*sigma_i_minus_1)
            else:
                alpha_i_minus_2 = 1 if i == 1 else sd_pipe.scheduler.alphas_cumprod[timesteps[index + 2]].to(torch.float32)
                sigma_i_minus_2 = (1 - alpha_i_minus_2) ** 0.5
                alpha_i_minus_2 = alpha_i_minus_2 ** 0.5

                coef_x_i_minus_2 = (1/gamma)
                coef_x_i_minus_1 = alpha_i/alpha_i_minus_1-(1/gamma)*(alpha_i_minus_2/alpha_i_minus_1)
                coef_eps = (sigma_i - (alpha_i/alpha_i_minus_1)*sigma_i_minus_1)-(1/gamma)*(sigma_i_minus_2-(alpha_i_minus_2/alpha_i_minus_1)*sigma_i_minus_1)
                latent = coef_x_i_minus_2 * xis[-2] + coef_x_i_minus_1 * xis[-1] + coef_eps * noise_pred
            xis.append(latent)
    return xis[-1], xis[-2]

# import torch
# from cn_dm.test.adjoint_state import test_sd15, BDIA
# sd_pipe, clip_model, ae_model, trans = test_sd15.load_models(torch.float32)
# prompt = 'Girl with cat, symmetrical face, sharp focus, intricate details, soft lighting, detailed face, blur background'
# negative_prompt='lowres, error, cropped, worst quality, low quality, jpeg artifacts, out of frame, watermark, signature, deformed, ugly, mutilated, disfigured, text, extra limbs, face cut, head cut, extra fingers, extra arms, poorly drawn face, mutation, bad proportions, cropped head, malformed limbs, mutated hands, fused fingers, long neck'
# sd_params = {'prompt':prompt, 'negative_prompt':negative_prompt, 'seed':91254625325, 'guidance_scale':7.5, 'num_inference_steps':20, 'width':512, 'height':512}
# res = BDIA.rev_forward(sd_pipe, sd_params, latents=None)