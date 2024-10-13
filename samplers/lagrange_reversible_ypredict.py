import torch
from cn_dm.test.adjoint_state.test_sd15 import to_pil

def rev_forward(sd_pipe, sd_params, latents=None):

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
            if not prev_noise is None:
                print(i, abs(prev_noise - noise_pred).mean())
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

            n_mid = 10
            # if i > n_mid:
            #     alpha_t = sd_pipe.scheduler.alphas_cumprod[timesteps[i-1]].to(torch.float32)
            sigma_s = (1 - alpha_s)**0.5
            sigma_t = (1 - alpha_t)**0.5
            alpha_s = alpha_s**0.5
            alpha_t = alpha_t**0.5
            # print('#', i , ' sigma_t / alpha_t: ', sigma_t / alpha_t)
            # print('#', i, ' sigma_s / alpha_s - sigma_t / alpha_t: ', sigma_s / alpha_s - sigma_t / alpha_t)

            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            if i == 0 or i <= n_mid:
                latents = coef_xt * latents + coef_eps * noise_pred
            else:
                # calculate i-1
                alpha_p = sd_pipe.scheduler.alphas_cumprod[timesteps[i - 1]].to(torch.float32)
                sigma_p = (1 - alpha_p) ** 0.5
                alpha_p = alpha_p ** 0.5

                # calculate t
                t_p, t_t, t_s = alpha_p / sigma_p, alpha_t / sigma_t, alpha_s / sigma_s

                # calculate delta
                delta_1 = t_t - t_p
                delta_2 = t_s - t_t
                delta_3 = t_s - t_p

                # calculate coef
                coef_1 = -(sigma_s*sigma_t)/alpha_t *(delta_2*delta_3)/delta_1
                coef_2 = (delta_2/delta_1)**2*sigma_s / sigma_p
                coef_3 = (sigma_s*delta_2*delta_3/delta_1)*(1.0/alpha_t-(delta_2 - delta_1)/(delta_1*delta_2*sigma_t))

                # iterate
                latents = coef_1 * noise_pred + coef_2 * xis[-2] + coef_3 * xis[-1]

                # coef_xt = coef_xt - alpha_p / alpha_t
                # coef_eps_2 = sigma_p - sigma_t * alpha_p / alpha_t
                # coef_eps = coef_eps - coef_eps_2
                # print(latents.shape)
                # print(xis[-1].shape)
                # assert torch.equal(latents, xis[-1])
                # latents = xis[-2] + coef_xt * xis[-1] + coef_eps * noise_pred
            xis.append(latents)
            prev_noise = noise_pred.clone()
    return to_pil(xis[-1], sd_pipe)

# import torch
# from cn_dm.test.adjoint_state import test_sd15, langrange_reversible
# sd_pipe, clip_model, ae_model, trans = test_sd15.load_models(torch.float32)
# prompt = 'Girl with cat, symmetrical face, sharp focus, intricate details, soft lighting, detailed face, blur background'
# negative_prompt='lowres, error, cropped, worst quality, low quality, jpeg artifacts, out of frame, watermark, signature, deformed, ugly, mutilated, disfigured, text, extra limbs, face cut, head cut, extra fingers, extra arms, poorly drawn face, mutation, bad proportions, cropped head, malformed limbs, mutated hands, fused fingers, long neck'
# sd_params = {'prompt':prompt, 'negative_prompt':negative_prompt, 'seed':91254625325, 'guidance_scale':7.5, 'num_inference_steps':20, 'width':512, 'height':512}
# res = langrange_reversible.rev_forward(sd_pipe, sd_params, latents=None)