import torch
from samplers.test_sd15 import to_pil

def rev_forward(sd_pipe, sd_params, latents=None, p = 0.93):

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
    x_latents = latents

    xis = []
    yis = []
    do_classifier_free_guidance = guidance_scale > 1.0
    if x_latents is None:
        shape = (1, 4, 64, 64)
        x_latents = torch.randn(shape, generator=None, device='cuda', dtype=dtype)
        print('latents are None')

    y_latents = x_latents
    yis.append(y_latents)
    xis.append(x_latents)
    prev_noise = None
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # print('###', i)
            latent_model_input = torch.cat([y_latents] * 2) if do_classifier_free_guidance else y_latents
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
            # print('new#', i , ' sigma_t / alpha_t: ', sigma_t / alpha_t)
            # print('new#', i, ' sigma_s / alpha_s - sigma_t / alpha_t: ', sigma_s / alpha_s - sigma_t / alpha_t)

            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt

            x_inter = coef_xt * x_latents + coef_eps * noise_pred
            latent_model_input = torch.cat([x_inter] * 2) if do_classifier_free_guidance else x_inter
            noise_pred = sd_pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            y_inter = coef_xt * y_latents + coef_eps * noise_pred

            x_latents = p * x_inter + (1.0 - p) * y_inter
            y_latents = p * y_inter + (1.0 - p) * x_latents
            xis.append(x_latents)
            yis.append(y_latents)
            prev_noise = noise_pred.clone()
    return to_pil(xis[-1], sd_pipe)

def intermediate_to_latent(sd_pipe, sd_params, x_intermediate=None, y_intermediate =None, p = 0.93, freeze_step = 0):

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
    x_latents = x_intermediate
    y_latents = y_intermediate
    xis = []
    yis = []
    do_classifier_free_guidance = guidance_scale > 1.0
    if x_latents is None:
        shape = (1, 4, 64, 64)
        x_latents = torch.randn(shape, generator=None, device='cuda', dtype=dtype)
        print('latents are None')
    if y_latents is None:
        shape = (1, 4, 64, 64)
        y_latents = torch.randn(shape, generator=None, device='cuda', dtype=dtype)
        print('latents are None')

    yis.append(y_latents)
    xis.append(x_latents)
    prev_noise = None
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            if i < freeze_step:
                continue
            # print('###', i)
            latent_model_input = torch.cat([y_latents] * 2) if do_classifier_free_guidance else y_latents
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

            x_inter = coef_xt * x_latents + coef_eps * noise_pred
            latent_model_input = torch.cat([x_inter] * 2) if do_classifier_free_guidance else x_inter
            noise_pred = sd_pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            y_inter = coef_xt * y_latents + coef_eps * noise_pred

            x_latents = p * x_inter + (1.0 - p) * y_inter
            y_latents = p * y_inter + (1.0 - p) * x_latents
            xis.append(x_latents)
            yis.append(y_latents)
            prev_noise = noise_pred.clone()
    return xis[-1], yis[-1]

def latent_to_intermediate(sd_pipe, sd_params, latent=None, p = 0.93, freeze_step = 0):

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
    x_latents = latent

    xis = []
    yis = []
    do_classifier_free_guidance = guidance_scale > 1.0
    if x_latents is None:
        shape = (1, 4, 64, 64)
        x_latents = torch.randn(shape, generator=None, device='cuda', dtype=dtype)
        print('latents are None')

    y_latents = x_latents
    yis.append(y_latents)
    xis.append(x_latents)
    prev_noise = None
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            if i >= num_inference_steps - freeze_step:
                continue
            # print('###', i)
            index = num_inference_steps - i - 1
            # time = timesteps[index + 1] if index < num_inference_steps - 1 else 1
            time = timesteps[index]
            if index < num_inference_steps - 1:
                alpha_i = sd_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_i_minus_1 = sd_pipe.scheduler.alphas_cumprod[timesteps[index+1]].to(torch.float32)
            else:
                alpha_i = sd_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_i_minus_1 = 1

            sigma_i = (1 - alpha_i)**0.5
            sigma_i_minus_1 = (1 - alpha_i_minus_1)**0.5
            alpha_i = alpha_i**0.5
            alpha_i_minus_1 = alpha_i_minus_1**0.5

            a_i = alpha_i_minus_1 / alpha_i
            b_i = (sigma_i_minus_1 - a_i * sigma_i)

            y_inter = (y_latents - (1-p) * x_latents)/p
            x_inter = (x_latents - (1-p)*y_inter)/p


            latent_model_input = torch.cat([x_inter] * 2) if do_classifier_free_guidance else x_inter
            noise_pred = sd_pipe.unet(
                latent_model_input,
                time,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            y_latents = (y_inter - b_i * noise_pred)/a_i

            latent_model_input = torch.cat([y_latents] * 2) if do_classifier_free_guidance else y_latents
            noise_pred = sd_pipe.unet(
                latent_model_input,
                time,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            x_latents = (x_inter - b_i * noise_pred)/a_i

            xis.append(x_latents)
            yis.append(y_latents)
    return xis[-1], yis[-1]



# import torch
# from cn_dm.test.adjoint_state import test_sd15, langrange_reversible
# sd_pipe, clip_model, ae_model, trans = test_sd15.load_models(torch.float32)
# prompt = 'Girl with cat, symmetrical face, sharp focus, intricate details, soft lighting, detailed face, blur background'
# negative_prompt='lowres, error, cropped, worst quality, low quality, jpeg artifacts, out of frame, watermark, signature, deformed, ugly, mutilated, disfigured, text, extra limbs, face cut, head cut, extra fingers, extra arms, poorly drawn face, mutation, bad proportions, cropped head, malformed limbs, mutated hands, fused fingers, long neck'
# sd_params = {'prompt':prompt, 'negative_prompt':negative_prompt, 'seed':91254625325, 'guidance_scale':7.5, 'num_inference_steps':20, 'width':512, 'height':512}
# res = langrange_reversible.rev_forward(sd_pipe, sd_params, latents=None)