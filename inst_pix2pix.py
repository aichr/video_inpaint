import os
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
from diffusers.utils import make_image_grid
from prompts import PROMPTS_HEYGEN_DEMO

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config)

inp_img = "./data/frames/0001.png"
image = Image.open(inp_img)

ouptut_folder = "output_inst_pix2pix"
os.makedirs(ouptut_folder, exist_ok=True)

idx = 0
prompts = PROMPTS_HEYGEN_DEMO
strength = 0.99
guidance_scale = 8
for prompt in prompts:
    prompt = "make the woman wear " + prompt
    num_images = 4
    generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(num_images)]

    repainted = pipe(prompt, image=image, strength=strength,
                     guidance_scale=guidance_scale, num_images_per_prompt=4).images

    # image_list = [image, repainted]
    grid = make_image_grid(repainted, rows=1, cols=num_images)
    output_path = f"inst_pix2pix_{idx}.png"
    grid.save(os.path.join(ouptut_folder, output_path))
    idx += 1
