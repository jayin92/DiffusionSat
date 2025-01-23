import os
import torch

from diffusers import DDIMInverseScheduler

from PIL import Image
from PIL.Image import Image as PILImage
from diffusionsat import (
    SatUNet, DiffusionSatPipeline,
    metadata_normalize
)

from tqdm import tqdm
from torchvision import transforms as tvt

from typing import Tuple, List


null_metadata = metadata_normalize([0.0, 0.0, 0.0, 0.05, 2015, 2, 27]).tolist()

class DiffusionSatRefineIDU:
    def __init__(self, save_path, device="cuda:0"):
        self.device = device
        self.save_path = save_path
        self.model_path = '/data1/jayinnn/DiffusionSat/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64/'
        self.pipe, self.pipe_inv = self.load_pipe(device=device)
        os.makedirs(save_path, exist_ok=True)

    def img_to_latents(self, x, vae):
        x = 2. * x - 1.
        posterior = vae.encode(x).latent_dist
        latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def sample(self, pipe, start_latents, prompt, metadata, negative_prompt="", num_steps: int=50, strength: float= 1.0, guidance_scale=1):
        start_step = int(num_steps * (1.0 - strength))
        latents = start_latents.clone()
        pipe.scheduler.set_timesteps(num_steps)
        # text_embeddings = pipe._encode_prompt(
        #     prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        # )
        text_embeddings = pipe._encode_prompt(
            prompt, self.device, 1, True, negative_prompt
        )
        input_metadata = pipe.prepare_metadata(1, metadata, True, self.device, text_embeddings.dtype)
        for i in tqdm(range(start_step, num_steps)):
            t = pipe.scheduler.timesteps[i]

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = pipe.unet(
                latent_model_input, t, metadata=input_metadata, encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=None).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        images = pipe.decode_latents(latents)
        images = pipe.numpy_to_pil(images)

        return images, latents

    @torch.no_grad()
    def partial_ddim_inversion(self, img: PILImage, num_steps:int=50, strength=1.0, guidance_scale=1) -> PILImage:
        device = self.device
        vae = self.pipe_inv.vae
        img = tvt.ToTensor()(img)[None, ...]
        img = img.to(device)

        latents = self.img_to_latents(img, vae)

        _, inv_latents = self.sample(self.pipe_inv, latents, 
                                prompt="a fmow satellite image",
                                metadata=null_metadata,
                                num_steps=num_steps,
                                strength=strength,
                                guidance_scale=guidance_scale)
        image, _ = self.sample(self.pipe, inv_latents,
                        prompt="a fmow satellite image",
                        metadata=null_metadata,
                        num_steps=num_steps,
                        strength=strength,
                        guidance_scale=guidance_scale)

        return image[0]
        
    def load_pipe(self) -> Tuple[DiffusionSatPipeline, DiffusionSatPipeline]:
        device = self.device
        path = self.model_path
        unet = SatUNet.from_pretrained(path + 'checkpoint-150000', subfolder="unet", torch_dtype=torch.float32)
        pipe = DiffusionSatPipeline.from_pretrained(
            path, unet=unet,
            torch_dtype=torch.float32)
        pipe = pipe.to(device)

        unet_inv = SatUNet.from_pretrained(path + 'checkpoint-150000', subfolder="unet", torch_dtype=torch.float32)
        scheduler_config = {
            'num_train_timesteps': 1000,
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'scaled_linear',
            'trained_betas': None,
            'clip_sample': False,
            'set_alpha_to_one': False,
            'steps_offset': 1,
            'prediction_type': 'v_prediction',
            'thresholding': False,
            'dynamic_thresholding_ratio': 0.995,
            'clip_sample_range': 1.0,
            'sample_max_value': 1.0
        }
        scheduler_inv = DDIMInverseScheduler(**scheduler_config)
        print(scheduler_inv.config)
        pipe_inv = DiffusionSatPipeline.from_pretrained(
            path, unet=unet_inv,
            torch_dtype=torch.float32)
        pipe_inv.scheduler = scheduler_inv
        pipe_inv = pipe_inv.to(device)

        return pipe, pipe_inv
    
    @torch.no_grad()
    def run(self, imgs: List[PILImage], num_steps: int=50, strength=0.1, guidance_scale=1):
        refine_imgs = []
        for idx, img in enumerate(tqdm(imgs, desc="Refining images using DiffusionSat")):
            refine_img = self.partial_ddim_inversion(img, num_steps=num_steps, strength=strength, guidance_scale=guidance_scale)
            refine_img.save(os.path.join(self.save_path, '{0:05d}'.format(idx) + ".png"))
            refine_imgs.append(refine_img)

        return refine_imgs