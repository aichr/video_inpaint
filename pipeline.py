import os
import glob
from inpaint import sd_inpaint
from prompts import PROMPTS_FASHION


def main():
    input_img = "./data/frames/0001.png"
    input_mask = "./data/inpaint_mask/0001.png"

    input_frame_dir = "./data/frames"
    input_mask_dir = "./data/inpaint_mask"
    input_frames = sorted(glob.glob(os.path.join(input_frame_dir, "*.png")))
    input_masks = sorted(glob.glob(os.path.join(input_mask_dir, "*.png")))
    assert len(input_frames) == len(input_masks)

    prompts = PROMPTS_FASHION
    positive_prompt = "intricate details, beautiful, elegant, proportional hands, realistic, in the style of Serpieri, very detailed illustration"
    negative_prompt = "bad anatomy, deformed, ugly, disfigured, unnatural hands, disfigured hands, bad hands, bent fingers, extra joints"

    output_folder = "./output_inpaint"

    step = 20
    for i in range(0, len(input_frames), step):
        input_img = input_frames[i]
        input_mask = input_masks[i]
        key_output_folder = os.path.join(
            "./output_inpaint", os.path.basename(input_img).split(".")[0])
        os.makedirs(output_folder, exist_ok=True)

        sd_inpaint(input_img, input_mask, prompts, positive_prompt,
                   negative_prompt, output_folder=key_output_folder)


if __name__ == "__main__":
    main()
