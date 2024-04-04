import argparse
import os
import glob
from inpaint import sd_inpaint
from prompt_constants import PROMPTS_FASHION
from video_utils import get_fps, ffmpeg_concat, extract_audio, add_audio


def main(run_inpaint=False, run_ebsynth=False, run_ffmpeg=True):
    input_video = "./data/demo_input.mp4"
    input_frame_dir = "./data/frames"
    input_mask_dir = "./data/inpaint_mask"
    input_frames = sorted(glob.glob(os.path.join(input_frame_dir, "*.png")))
    input_masks = sorted(glob.glob(os.path.join(input_mask_dir, "*.png")))
    num_frames = len(input_frames)
    assert num_frames == len(input_masks)

    output_eb = "./output_ebsynth"
    output_inpaint = "./output_inpaint"

    step = 20
    if run_inpaint:
        prompts = PROMPTS_FASHION
        positive_prompt = "intricate details, beautiful, elegant, proportional hands, realistic, in the style of Serpieri, very detailed illustration"
        negative_prompt = "bad anatomy, deformed, ugly, disfigured, unnatural hands, disfigured hands, bad hands, bent fingers, extra joints"

        for i in range(0, len(input_frames), step):
            input_img = input_frames[i]
            input_mask = input_masks[i]
            sub_dir = os.path.basename(input_img).split(".")[0]
            key_output_folder = os.path.join(output_inpaint, sub_dir)
            os.makedirs(key_output_folder, exist_ok=True)

            sd_inpaint(input_img, input_mask, prompts, positive_prompt,
                       negative_prompt, output_folder=key_output_folder)

    if run_ebsynth:
        # select the best frames, currently hard-coded
        key_frames = {
            "0001": "output_inpaint/0001/inpainting_24.png",
            "0021": "output_inpaint/0021/inpainting_23.png",
            "0041": "output_inpaint/0041/inpainting_0.png",
            "0061": "output_inpaint/0061/inpainting_9.png",
            "0081": "output_inpaint/0081/inpainting_25.png",
            "0101": "output_inpaint/0101/inpainting_10.png",
            "0121": "output_inpaint/0121/inpainting_15.png",
        }

        os.makedirs(output_eb, exist_ok=True)
        for i in range(0, len(input_frames), step):
            key = os.path.basename(input_frames[i]).split(".")[0]

            for j in range(i+1, i + step+1):
                if j >= num_frames:
                    break
                # -searchvoteiters 12 -patchmatchiters 6"
                cmd = f"ebsynth/bin/ebsynth -style {key_frames[key]} -guide {input_frames[i]} {input_frames[j]} -output {output_eb}/{j:04d}.png"
                print(cmd)
                os.system(cmd)

    if run_ffmpeg:
        # combine frames
        output_video = "./output_ebsynth.mp4"
        fps = get_fps(input_video)
        ffmpeg_concat(input_file=f"{output_eb}/%04d.png",
                      output_file=output_video, fps=fps)

        with_audio = True
        if with_audio:
            audio_clip = extract_audio(input_video)
            add_audio(output_video, audio_clip,
                      output_video.replace(".mp4", "_final.mp4"))

        # add original audio to the video


if __name__ == "__main__":
    # create a argument parser for main
    parser = argparse.ArgumentParser()
    parser.add_argument("-ri", action="store_true", help="run inpaint")
    parser.add_argument("-re", action="store_true", help="run ebsynth")
    parser.add_argument("-rf", action="store_true", help="run ffmpeg")

    args = parser.parse_args()
    main(run_inpaint=args.ri, run_ebsynth=args.re, run_ffmpeg=args.rf)
