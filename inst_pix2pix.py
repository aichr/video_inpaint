import os
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
from diffusers.utils import make_image_grid
from prompts import PROMPTS_HEYGEN_DEMO

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

inp_img = "./data/frames/0001.png"
image = Image.open(inp_img)

ouptut_folder = "output_inst_pix2pix"
os.makedirs(ouptut_folder, exist_ok=True)

idx = 0
prompts = PROMPTS_HEYGEN_DEMO
for prompt in prompts:
    prompt = "wearing " + prompt
    repainted = pipe(prompt, image=image).images[0]

    image_list = [image, repainted]
    grid = make_image_grid(image_list, rows=1, cols=len(image_list))
    output_path = f"inst_pix2pix_{idx}.png"
    grid.save(os.path.join(ouptut_folder, output_path))
    idx += 1