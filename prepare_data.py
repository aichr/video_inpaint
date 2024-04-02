import os
import glob
from video_utils import run_zoe_segment, face_detect, mask_bool_to_image, extract_frames, crop_func, create_zoe_model


def prepare(filename):
    # Extract frames from video
    l, r, t, b = [608, 608, 56, 0]
    new_w, new_h = None, None

    extract_frames(
        filename,
        output_folder="./data/frames",
        crop_func=lambda frame: crop_func(frame, l, r, t, b, new_w, new_h),
    )

    # Run Zoe segmentation on the frames
    input_folder = "./data/frames"
    zoe_output_dir = "./data/frames_zoe"
    inpaint_mask_output_dir = "./data/inpaint_mask"
    os.makedirs(zoe_output_dir, exist_ok=True)
    os.makedirs(inpaint_mask_output_dir, exist_ok=True)

    # Detect face in the frame
    # list all .png files

    input_images = sorted(glob.glob(os.path.join(input_folder, "*.png")))

    zoe = create_zoe_model()

    for input_img in input_images:
        x, y, w, h = face_detect(input_img, border=0)
        zoe_mask_bool = run_zoe_segment(
            input_img, zoe=zoe, output_folder=zoe_output_dir, skip_existing=False)[0]

        # only keep mask below the face
        img_base = os.path.basename(input_img)
        zoe_mask_bool[:y+h, :] = False
        mask_bool_to_image(zoe_mask_bool, os.path.join(
            inpaint_mask_output_dir, img_base))


    """ optionally use face-parsing to get fine-grained face masks
    # Get face mask using face parsing
    from face_parsing.test import get_face_mask
    face_mask = get_face_mask(face_output_path, respth="./", cp="./face_parsing/79999_iter.pth")

    frame = cv2.imread(input_img)
    full_face_mask = np.zeros(frame.shape[:2], dtype=bool)
    full_face_mask[y:y+h, x:x+w] = face_mask
    mask_bool_to_image(full_face_mask, "full_face_mask.png")

    # remove face mask from zoe mask
    zoe_mask_bool[full_face_mask] = False
    mask_bool_to_image(zoe_mask_bool, "final_mask.png")
    """


if __name__ == "__main__":
    filename = "./data/demo_input.mp4"
    prepare(filename)
