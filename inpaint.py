import os
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import numpy as np
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image, make_image_grid
from PIL import Image
from prompts import PROMPTS_FASHION, PROMPTS_HEYGEN_DEMO


def make_inpaint_condition(init_image, mask_image):
    init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

    assert init_image.shape[0:1] == mask_image.shape[0:
                                                     1], "image and image_mask must have the same image size"
    init_image[mask_image > 0.5] = -1.0  # set as masked pixel
    init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image)
    return init_image


def sd_inpaint(input_img: str, input_mask: str, prompt: str, negative_prompt: str, output_folder: str = None):

    init_image = load_image(input_img)
    mask_image = load_image(input_mask)

    assert init_image.size == mask_image.size
    w, h = init_image.size

    ####################### param block #######################
    use_controlnet = True
    use_openpose = True
    # a high strength value means more noise is added to an image and the denoising process takes longer, but you’ll get higher quality images that are more different from the base image
    # a low strength value means less noise is added to an image and the denoising process is faster, but the image quality may not be as great and the generated image resembles the base image more
    strength = 0.99   # [0, 1]
    # a high guidance_scale value means the prompt and generated image are closely aligned, so the output is a stricter interpretation of the prompt
    # a low guidance_scale value means the prompt and generated image are more loosely aligned, so the output may be more varied from the prompt
    guidance_scale = 8
    apply_ovelay = False
    infer_steps = 30
    ##########################################################

    if use_controlnet:
        if use_openpose:
            # openpose
            controlnet = ControlNetModel.from_pretrained(
                "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16)
            pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
            pipeline.scheduler = UniPCMultistepScheduler.from_config(
                pipeline.scheduler.config)

        else:
            # load ControlNet
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16")

            # pass ControlNet to the pipeline
            pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16"
            )
    else:
        # model = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
        model = "runwayml/stable-diffusion-inpainting"
        pipeline = AutoPipelineForInpainting.from_pretrained(
            model, torch_dtype=torch.float16)

    pipeline.enable_model_cpu_offload()
    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    pipeline.enable_xformers_memory_efficient_attention()

    if use_controlnet:
        if use_openpose:
            from controlnet_aux import OpenposeDetector

            openpose = OpenposeDetector.from_pretrained(
                'lllyasviel/ControlNet')

            pose_image = openpose(
                init_image, include_body=True, include_hand=False, include_face=False)
            pose_image = pose_image.resize((init_image.size), Image.NEAREST)

            repainted_image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image,
                                       mask_image=mask_image, num_inference_steps=infer_steps, height=h, width=w,
                                       strength=strength, guidance_scale=guidance_scale, control_image=pose_image).images[0]
            image_list = [init_image, mask_image, pose_image, repainted_image]

        else:
            control_image = make_inpaint_condition(init_image, mask_image)
            repainted_image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image,
                                       mask_image=mask_image, num_inference_steps=infer_steps, height=h, width=w,
                                       strength=strength, guidance_scale=guidance_scale, control_image=control_image,
                                       generator=torch.Generator(device="cuda")).images[0]
            image_list = [init_image, mask_image, Image.fromarray(
                np.uint8(control_image[0][0])).convert('RGB'), repainted_image]
    else:
        repainted_image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image,
                                   mask_image=mask_image, num_inference_steps=infer_steps, height=h, width=w,
                                   strength=strength, guidance_scale=guidance_scale).images[0]
        image_list = [init_image, mask_image, repainted_image]

    if apply_ovelay:
        unmasked_unchanged_image = pipeline.image_processor.apply_overlay(
            mask_image, init_image, repainted_image)
        image_list.append(unmasked_unchanged_image)

    grid = make_image_grid(image_list, rows=1, cols=len(image_list))

    # saving images
    prompt_str = prompt.replace(", ", ",").replace(" ", "_")[:20]
    output_name = "inpainting_" + prompt_str + ".png"
    grid_name = "grid_" + output_name
    if output_folder is not None:
        output_name = os.path.join(output_folder, output_name)
        grid_name = os.path.join(output_folder, grid_name)
    repainted_image.save(output_name)
    grid.save(grid_name)
    print(f"Saved inpainting result to {output_name}")


if __name__ == "__main__":
    input_img = "./data/frames/0001.png"
    input_mask = "./data/inpaint_mask/0001.png"

    prompts = PROMPTS_FASHION + PROMPTS_HEYGEN_DEMO
    positive_prompt = "intricate details, beautiful, elegant, proportional hands, realistic, in the style of Serpieri, very detailed illustration"
    negative_prompt = "bad anatomy, deformed, ugly, disfigured, unnatural hands, disfigured hands, bad hands, bent fingers, extra joints"

    output_folder = "./output_inpaint"
    os.makedirs(output_folder, exist_ok=True)
    for prompt in prompts:
        sd_inpaint(input_img, input_mask, prompt+","+positive_prompt,
                   negative_prompt, output_folder=output_folder)
