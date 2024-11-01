import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
# __dir__ = os.path.dirname(os.path.abspath('adjoint_state'))
sys.path = [os.path.abspath(os.path.join(__dir__, '../../../libs'))] + sys.path
sys.path = [os.path.abspath(os.path.join(__dir__, '../../../libs/sd_scripts'))] + sys.path

env_root = '.'
os.environ['HF_HOME'] = os.path.join(env_root, '.cache')
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(env_root, '.cache/hub')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(env_root, '.cache/transformers')
os.environ['TORCH_HOME'] = os.path.join(env_root, '.cache/torch')
os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(env_root, '.cache/torch_extensions')
os.environ['PYTORCH_KERNEL_CACHE_PATH'] = os.path.join(env_root, '.cache/torch')
os.environ['XDG_CACHE_HOME'] = os.path.join(env_root, '.cache/hub')
os.environ['TRITON_CACHE_DIR'] = os.path.join(env_root, '.cache/triton/autotune')

import torch
import diffusers
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms

from samplers.utils import PipelineLike, AestheticMLP
from PIL import Image
from functools import partial

def load_models(dtype=torch.float16):
    device = 'cuda'
    # sd
    model_path = 'xxxxx/stable-diffusion-v1-5'
    sd_15 = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)


    sche = DDIMScheduler(beta_end=0.012, beta_start=0.00085, beta_schedule='scaled_linear', clip_sample=False, timestep_spacing='linspace', set_alpha_to_one=False)

    sd_pipe = PipelineLike(device = device, vae = sd_15.vae, text_encoder = sd_15.text_encoder, tokenizer = sd_15.tokenizer, unet = sd_15.unet, scheduler = sche)
    sd_pipe.vae.to(device)
    sd_pipe.text_encoder.to(device)
    sd_pipe.unet.to(device)
    print('loaded')
    ae_model = None
    clip_model = None
    trans = None
    return sd_pipe, clip_model, ae_model, trans

def load_models_SD2_base(dtype=torch.float16):
    device = 'cuda'
    # sd
    model_path = 'xxxxxx/stable-diffusion-2-base'
    sd_2 = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)

    sche = DDIMScheduler(beta_end=0.012, beta_start=0.00085, beta_schedule='scaled_linear', clip_sample=False, timestep_spacing='linspace', set_alpha_to_one=False)

    sd_pipe = PipelineLike(device = device, vae = sd_2.vae, text_encoder = sd_2.text_encoder, tokenizer = sd_2.tokenizer, unet = sd_2.unet, scheduler = sche)
    sd_pipe.vae.to(device)
    sd_pipe.text_encoder.to(device)
    sd_pipe.unet.to(device)
    print('loaded')
    ae_model = None
    trans = None
    clip_model = None
    return sd_pipe, clip_model, ae_model, trans

def aes_trans(img, device):
    # img \in [0, 1]ƒ
    a = torch.nn.functional.interpolate(img, 224, mode='bicubic')
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    mean = mean[..., None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    std = std[..., None, None]
    a = (a - mean) / std
    return a.to(img.dtype)

def aes_score(aes_img, device, clip_model, ae_model):
    feat = clip_model.get_image_features(aes_img)
    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    prediction = ae_model(feat.float())
    return prediction

def get_score(img, device, clip_model, ae_model):
    # img \in [0, 1]
    aes_img = aes_trans(img, device)
    pred = aes_score(aes_img, device, clip_model, ae_model)
    return pred

def test_aesthetic_score(trans, clip_model, ae_model, device, dtype):


    for f in files:
        img = Image.open(os.path.join(root, f))
        with torch.no_grad():
            imgt = trans(img).unsqueeze(0).to(device, dtype=dtype)
            feat = clip_model.get_image_features(imgt)
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
            prediction = ae_model(feat.float())
        print(f, prediction)

def decode_vae(vae, latents):
    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    do_denormalize = [True]
    image = vae_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)
    return image

def get_aesthetic_score(img, clip_model, ae_model, trans, device, dtype):
    with torch.no_grad():
        imgt = trans(img).unsqueeze(0).to(device, dtype=dtype)
        feat = clip_model.get_image_features(imgt)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        prediction = ae_model(feat.float())
    return prediction

def explicit_euler(prompt, negative_prompt, sd_pipe, clip_model, ae_model, trans, device='cuda', dtype=torch.float32, guidance_scale=7.5, seed=91254625325, num_inference_steps=20, height=512, width=512, ode_type=2):
    do_classifier_free_guidance = guidance_scale > 1.0
    torch.manual_seed(seed)

    prompt_embeds = sd_pipe._encode_prompt(
        prompt,
        device,
        do_classifier_free_guidance,
        negative_prompt
    )
    sd_pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = sd_pipe.scheduler.timesteps
    num_channels_latents = sd_pipe.unet.config.in_channels
    shape = (1, num_channels_latents, height // sd_pipe.vae_scale_factor, width // sd_pipe.vae_scale_factor)
    latents = torch.randn(shape, generator=None, device=device, dtype=dtype)
    print(timesteps)

    xis = [latents]

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            print(i)
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = sd_pipe.scheduler.scale_model_input(latent_model_input, t)

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
                alpha_s = sd_pipe.scheduler.alphas_cumprod[timesteps[i+1]].to(torch.float32)
                alpha_s = alpha_s**0.5
                alpha_t = sd_pipe.scheduler.alphas_cumprod[t].to(torch.float32)
                alpha_t = alpha_t**0.5
            else:
                alpha_s = 1
                alpha_t = sd_pipe.scheduler.alphas_cumprod[t].to(torch.float32)
                alpha_t = alpha_t**0.5
            # print(alpha_t, alpha_s)
            sigma_s = (1 - alpha_s**2)**0.5
            sigma_t = (1 - alpha_t**2)**0.5
            # semi-expicit euler coef
            coef_xt, coef_eps = [], []
            log_coef = torch.log(alpha_t / alpha_s)
            coef_xt.append(1 - log_coef)
            coef_eps.append(log_coef / sigma_t)

            # expicit euler coef
            coef_xt.append(1 - torch.log(alpha_t / alpha_s))
            coef_eps.append((sigma_s**2 + sigma_t**2)/(2 * sigma_t) - coef_xt[-1] * sigma_t)

            # ddim
            coef_xt.append(alpha_s / alpha_t)
            coef_eps.append(sigma_s - sigma_t * alpha_s / alpha_t)
                
            print(coef_xt)
            print(coef_eps)
            latents = coef_xt[ode_type] * latents + coef_eps[ode_type] * noise_pred
            xis.append(latents)

        image = sd_pipe.vae.decode(latents / sd_pipe.vae.config.scaling_factor, return_dict=False)[0]

        # do_denormalize = [True]
        # image = sd_pipe.vae_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)
        image = image * 0.5 + 0.5

        return image, xis, prompt_embeds

def explicit_euler_forward(xis, lambs, prompt_embeds, sd_pipe, clip_model, ae_model, guidance_scale=7.5, strength=0.1):
    do_classifier_free_guidance = guidance_scale > 1.0
    init_step = len(lambs)
    n = len(xis)
    latents = xis[n - init_step] + lambs[-1] * strength
    timesteps = sd_pipe.scheduler.timesteps[n - 1 - init_step:]

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            print(i)
            latents = explicit_euler_step(i, t, latents, sd_pipe, prompt_embeds, timesteps, guidance_scale)

        image = sd_pipe.vae.decode(latents / sd_pipe.vae.config.scaling_factor, return_dict=False)[0]

        # do_denormalize = [True]
        # image = sd_pipe.vae_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)
        image = image * 0.5 + 0.5
        print(get_score(image, 'cuda', clip_model, ae_model))

    return image

def explicit_euler_backward(xis, prompt_embeds, sd_pipe, clip_model, ae_model, max_i=5, guidance_scale=7.5):
    do_classifier_free_guidance = guidance_scale > 1.0
    device = 'cuda'
    x0 = xis[-1]
    x0.requires_grad=True
    image = sd_pipe.vae.decode(x0 / sd_pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = image * 0.5 + 0.5
    pred = get_score(image, device, clip_model, ae_model)
    print(pred)
    lamb0 = torch.autograd.grad(pred, (x0,))[0]
    timesteps = sd_pipe.scheduler.timesteps.flip(0)
    lambs = [lamb0]
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            next_t = timesteps[i+1]
            if i == 0:
                alpha_s = 1
                alpha_t = sd_pipe.scheduler.alphas_cumprod[t].to(torch.float32)
            else:
                alpha_s = sd_pipe.scheduler.alphas_cumprod[t-1].to(torch.float32)
                alpha_t = sd_pipe.scheduler.alphas_cumprod[t].to(torch.float32)
            alpha_s = alpha_s**0.5
            alpha_t = alpha_t**0.5

            sigma_s = (1 - alpha_s**2)**0.5
            sigma_t = (1 - alpha_t**2)**0.5
            ds = (sigma_t / alpha_t - sigma_s / alpha_s)

            cur_x = xis[i]
            next_x = xis[i+1]

            cur_lam = lambs[i]

            with torch.enable_grad():
                next_x.requires_grad=True
                latent_model_input = torch.cat([next_x] * 2) if do_classifier_free_guidance else next_x
                noise_pred = sd_pipe.unet(
                    latent_model_input,
                    next_t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                prod = (noise_pred * cur_lam).sum()
                g = torch.autograd.grad(prod, (next_x))[0]
                next_x.requires_grad = False
                lambs.append(cur_lam - ds * g)

            if i >= max_i:
                break

    return lambs

def explicit_euler_step(i, t, latents, sd_pipe, prompt_embeds, timesteps, guidance_scale):
    do_classifier_free_guidance = guidance_scale > 1.0
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

    if i < len(timesteps) - 1:
        alpha_s = sd_pipe.scheduler.alphas_cumprod[timesteps[i+1]].to(torch.float32)
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
    latents = coef_xt * latents + coef_eps * noise_pred

    return latents

def explicit_euler_backward_step(i, xis, cur_lam, sd_pipe, prompt_embeds, timesteps, guidance_scale):
    do_classifier_free_guidance = guidance_scale > 1.0
    # cur_x = xis[-(i+1)]
    next_x = xis[-(i+2)]
    next_t = timesteps[-(i+1)]
    alpha_t = sd_pipe.scheduler.alphas_cumprod[next_t].to(torch.float32)
    alpha_s = 1 if i==0 else sd_pipe.scheduler.alphas_cumprod[timesteps[-i]].to(torch.float32)
    sigma_s = (1 - alpha_s)**0.5
    sigma_t = (1 - alpha_t)**0.5
    alpha_s = alpha_s**0.5
    alpha_t = alpha_t**0.5
    ds = (sigma_t / alpha_t - sigma_s / alpha_s)

    with torch.enable_grad():
        next_x.requires_grad = True
        latent_model_input = torch.cat([next_x] * 2) if do_classifier_free_guidance else next_x
        noise_pred = sd_pipe.unet(
            latent_model_input,
            next_t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        prod = (noise_pred * cur_lam).sum()
        g = torch.autograd.grad(prod, (next_x))[0]
        next_x.requires_grad = False

    return cur_lam - ds * g

def process_schemes(schemes):
    if schemes is None:
        res = [{'i':10, 'iter':1, 'adj_strength':10}, {'i':5, 'iter':2, 'adj_strength':10}]
        # res = [{'i':5, 'iter':1, 'adj_strength':10}]
    else:
        res = schemes
    return res

def adj_forward(sd_pipe, forward_param, latents=None):
    timesteps = forward_param['timesteps']
    dtype = forward_param['dtype']
    guidance_scale = forward_param['guidance_scale']
    prompt_embeds = forward_param['prompt_embeds']
    adj_i = forward_param['adj_i'] if not forward_param['adj_i'] is None else len(timesteps)
    xis = []
    do_classifier_free_guidance = guidance_scale > 1.0
    if latents is None:
        shape = (1, 4, 64, 64)
        latents = torch.randn(shape, generator=None, device='cuda', dtype=dtype)
        print('latents are None')
    else:
        print('latents shape:', latents.shape)
    xis.append(latents)
    with torch.no_grad():
        for i, t in enumerate(timesteps[-adj_i:]):
            latents = explicit_euler_step(i, t, latents, sd_pipe, prompt_embeds, timesteps[-adj_i:], guidance_scale)
            xis.append(latents)

    return xis

def adj_backward(sd_pipe, xis, backward_param):
    adj_i = backward_param['adj_i']
    timesteps = backward_param['timesteps']
    cur_lam = backward_param['cur_lam']
    prompt_embeds = backward_param['prompt_embeds']
    guidance_scale = backward_param['guidance_scale']
    for i in range(adj_i):
        cur_lam = explicit_euler_backward_step(i, xis, cur_lam, sd_pipe, prompt_embeds, timesteps, guidance_scale)
    return cur_lam

def to_pil(latents, sd_pipe):
    image = sd_pipe.vae.decode(latents / sd_pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = torch.clamp(image * 0.5 + 0.5, 0, 1)
    return transforms.ToPILImage()(image[0])

def to_tensor_image(latents, sd_pipe):
    image = sd_pipe.vae.decode(latents / sd_pipe.vae.config.scaling_factor, return_dict=False)[0]
    return torch.clamp(image * 0.5 + 0.5, 0, 1)

def symp_adjoint(models, sd_params, lam_func, schemes=None):
    sd_pipe = models['sd_pipe']
    # clip_model = models['clip_model']
    # ae_model = models['ae_model']

    schemes = process_schemes(schemes)
    prompt = sd_params['prompt']
    negative_prompt = sd_params['negative_prompt']
    seed = sd_params['seed']
    guidance_scale = sd_params['guidance_scale']
    num_inference_steps = sd_params['num_inference_steps']
    width = sd_params['width']
    height = sd_params['height']

    prompt_embeds = sd_pipe._encode_prompt(
        prompt,
        'cuda',
        guidance_scale > 1.0,
        negative_prompt
    )

    torch.manual_seed(seed)

    sd_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')

    forward_param = {}
    forward_param['timesteps'] = sd_pipe.scheduler.timesteps
    forward_param['dtype'] = torch.float32
    forward_param['guidance_scale'] = guidance_scale
    forward_param['prompt_embeds'] = prompt_embeds
    forward_param['adj_i'] = None
    xis = adj_forward(sd_pipe, forward_param, latents=None)
    ori_img = to_pil(xis[-1], sd_pipe)

    backward_param = {}
    backward_param['prompt_embeds'] = prompt_embeds
    backward_param['guidance_scale'] = guidance_scale
    backward_param['timesteps'] = sd_pipe.scheduler.timesteps

    for sch_idx in range(len(schemes)):
        print('sch_idx', sch_idx)
        sch = schemes[sch_idx]
        adj_i = sch['i']
        adj_iter = sch['iter']
        adj_strength = sch['adj_strength']

        for i in range(adj_iter):
            print(i, 'lambda')
            # lambda
            with torch.enable_grad():
                cur_latent = xis[-1]
                cur_latent.requires_grad = True
                cur_img = to_tensor_image(cur_latent, sd_pipe)
                pred = lam_func(cur_img)
                cur_lam = torch.autograd.grad(pred, (cur_latent,))[0]
                cur_latent.requires_grad = False
                print('pred', pred)
            print(i, 'backward')
            # backward
            backward_param['adj_i'] = adj_i
            backward_param['cur_lam'] = cur_lam
            res_lam = adj_backward(sd_pipe, xis, backward_param)
            latents = xis[-(adj_i+1)] + adj_strength * res_lam
            print(i, 'forward')
            # forward
            forward_param['adj_i'] = adj_i
            xis = adj_forward(sd_pipe, forward_param, latents=latents)

    res_img = to_pil(xis[-1], sd_pipe)
    pred = lam_func(to_tensor_image(xis[-1], sd_pipe))
    print('res_pred', pred)
    return xis, ori_img, res_img

def center_crop(im):
    width, height = im.size   # Get dimensions
    min_dim = min(width, height)
    left = (width - min_dim)/2
    top = (height - min_dim)/2
    right = (width + min_dim)/2
    bottom = (height + min_dim)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im

def load_im_into_format_from_path(im_path):
    return center_crop(Image.open(im_path)).resize((512,512))

def pil_to_latents(pil_image, sd_pipe):
    image_tensor = transforms.Compose([transforms.PILToTensor()])(pil_image).to('cuda')
    # print(image_tensor.shape)
    if image_tensor.shape[0] == 3:
        pass
    elif image_tensor.shape[0] == 1:
        image_tensor = image_tensor.repeat(3, 1, 1)
    else:
        raise ValueError("")
    # print('11',image_tensor)
    image_tensor = image_tensor / 255.0
    image_tensor = (image_tensor - 0.5) / 0.5
    # print('image_tensor.shape = ',image_tensor.shape)
    with torch.no_grad():
        latents = sd_pipe.vae.encode(image_tensor.unsqueeze(0), return_dict=False)[0].sample()
        # print('latent.shape = ',latents.shape)
    latents = latents * sd_pipe.vae.config.scaling_factor
    return latents
