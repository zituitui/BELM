import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path = [os.path.abspath(os.path.join(__dir__, '../../../libs'))] + sys.path
sys.path = [os.path.abspath(os.path.join(__dir__, '../../../libs/sd_scripts'))] + sys.path

from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
)

import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import AttnProcessor2_0

class PipelineLike:
    def __init__(
        self,
        device,
        vae: AutoencoderKL,
        text_encoder,
        tokenizer,
        unet: UNet2DConditionModel,
        scheduler,
    ):

        super().__init__()
        self.device = device
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # use sdpa
        self.unet.set_attn_processor(AttnProcessor2_0())

    def _encode_prompt(
        self,
        prompt,
        device,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        batch_size = 1
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        prompt = [prompt]
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape

        if do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [""]
            else:
                uncond_tokens = [negative_prompt]

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(device=device)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator = None,
        latents = None,
        output_type = "pil",
        return_latents: bool = False,
        seed = 91254625325,
        **kwargs
    ):
        device = self.device
        dtype = torch.float16
        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1.0

        torch.manual_seed(seed)

        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            do_classifier_free_guidance,
            negative_prompt
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels

        shape = (1, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        do_denormalize = [True]
        image = self.vae_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        return image

class AestheticMLP(torch.nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)