import glob
import os
import cv2
import numpy as np
from PIL import Image
import torch
import cv2
import imageio
from moviepy.editor import VideoFileClip, AudioClip


def apply_mask(image, mask, grayscale=255):
    # Ensure mask and image have the same shape
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask must have the same dimensions")

    canvas = np.ones_like(image) * grayscale
    overlay = np.where(mask[:, :, np.newaxis], image, canvas)

    return overlay


def zoe_segment(image_path, zoe, foreground_threshold=1.5, bg_color=225, return_mask=False):  # Zoe_N
    # model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # zoe = model_zoe_n.to(DEVICE)

    image = Image.open(image_path).convert("RGB")  # load

    depth_tensor = zoe.infer_pil(
        image, output_type="tensor")  # as torch tensor
    mask = depth_tensor < foreground_threshold
    mask = mask.detach().cpu().numpy()

    image = np.array(image)
    filter_image = apply_mask(image, mask, grayscale=bg_color)
    if return_mask:
        return filter_image, mask

    return filter_image


def video_down_sample(filename, down_factor=2):
    cap = cv2.VideoCapture(filename)

    # Get the video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the output video file and codec
    output_video_path = (
        os.path.splitext(filename)[0] + "_half" + os.path.splitext(filename)[1]
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for mp4 format
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # codec for H.264 format
    output = cv2.VideoWriter(output_video_path, fourcc,
                             fps, (width // 2, height // 2))

    # Process each frame and resize
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to half of its original size
        resized_frame = cv2.resize(
            frame, (width // down_factor, height // down_factor))

        # Write the resized frame to the output video
        output.write(resized_frame)

        # Show progress
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"Processed frame {frame_id}/{total_frames}")

    # Release the video capture and writer objects
    cap.release()
    output.release()

    print("Video resizing completed!")


def get_fps(video_path: str):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps


def crop_func(frame, l, r, t, b, new_w=None, new_h=None):
    H, W, C = frame.shape
    left = np.clip(l, 0, W)
    right = np.clip(W - r, left, W)
    top = np.clip(t, 0, H)
    bottom = np.clip(H - b, top, H)
    frame = frame[top:bottom, left:right]
    H, W, C = frame.shape

    if new_w is not None and new_h is not None:
        frame = cv2.resize(frame, (new_w, new_h))
    print(f"Frame shape: {frame.shape}")
    return frame


def extract_frames(filename, output_folder, down_factor=2, crop_func=None):
    """
    Given a mp4 video file, extract each frame and save it as a png file
    in the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(filename)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process each frame and resize
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if crop_func is not None:
            frame = crop_func(frame)

        # save frame to output folder
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_path = os.path.join(output_folder, f"{frame_id:04d}.png")

        cv2.imwrite(frame_path, frame)

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"Processed frame {frame_id}/{total_frames}")

    cap.release()


def video_overlay(video, input_folder, l, r, t, b):
    """
    overlay the images to the original video. Have to make sure the images are from the video.
    """
    # if not os.path.exists(output_folder):
    #    os.makedirs(output_folder)

    cap = cv2.VideoCapture(video)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    input_images = sorted(glob.glob(os.path.join(input_folder, "*.png")))

    # Process each frame and resize
    idx = 0
    while cap.isOpened() and idx < len(input_images):
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.open(input_images[idx])
        img = np.array(img)[:, :, :3]

        H, W, C = frame.shape
        left = np.clip(l, 0, W)
        right = np.clip(W - r, left, W)
        top = np.clip(t, 0, H)
        bottom = np.clip(H - b, top, H)

        frame[top:bottom, left:right, :] = img

        # TODO: blend in the two images so they appear more natural
        # Image.fromarray(frame).save(f"./{idx:04d}_overlay.png")

        # fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(idx)
        idx += 1

    cap.release()


def mask_bool_to_image(mask_bool, mask_name):
    mask = mask_bool[:, :, np.newaxis].astype(np.uint8) * 255
    mask = np.repeat(mask, 3, axis=2)
    Image.fromarray(mask).save(mask_name)


def create_zoe_model():
    # Zoe_N
    model_zoe_n = torch.hub.load(
        "./ZoeDepth", "ZoeD_N", source="local", pretrained=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)
    return zoe


def run_zoe_segment(input, output_folder, zoe=None, bg_color=100, skip_existing=True, foreground_threshold=1.5):
    """Segment the foreground using zoe depth model. The foreground is defined as pixels
    with the depth value less than the foreground_threshold.

    """
    # Get all png files in the directory
    if os.path.isdir(input):
        png_files = [file for file in os.listdir(
            input) if file.endswith(".png")]
        png_files = [os.path.join(input, file) for file in png_files]
    else:
        png_files = [input]

    if zoe is None:
        zoe = create_zoe_model()

    os.makedirs(output_folder, exist_ok=True)
    masks = []
    for idx, png in enumerate(png_files):
        filter_image_path = os.path.join(output_folder, os.path.basename(png))
        if skip_existing and os.path.exists(filter_image_path):
            continue
        print(f"run zoe_segment on {png}")
        filter_image, mask_bool = zoe_segment(
            image_path=png, zoe=zoe, foreground_threshold=foreground_threshold, bg_color=bg_color, return_mask=True)
        Image.fromarray(filter_image).save(filter_image_path)

        # convert mask to image
        mask_name = os.path.join(
            output_folder, f"{os.path.splitext(os.path.basename(png))[0]}_mask.png")
        mask_bool_to_image(mask_bool, mask_name)
        masks.append(mask_bool)
    return masks


def face_detect(frame, border=0, output_path=None):
    """
    Detect face in the frame and save the face to a file.
    Return the bounding box position of the face in (x, y, w, h).
    """
    if isinstance(frame, str):
        frame = cv2.imread(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for i, (x, y, w, h) in enumerate(faces):
        x -= border
        y -= border
        w += 2 * border
        h += 2 * border
        face = frame[y:y + h, x:x + w]

        if output_path is not None:
            cv2.imwrite(output_path, face)
            print(f"write to {output_path}")
    return x, y, w, h


def ffmpeg_concat(input_file="%04d.png", fps=25, output_file="output.mp4"):
    """Concatenate multiple video files into a single video file using ffmpeg."""
    crf = 5  # lower the better
    cmd = f"ffmpeg -framerate {fps} -i {input_file} -c:v libx264 -crf {crf} -pix_fmt yuv420p {output_file} -y"
    print(cmd)
    os.system(cmd)


def frame_to_video(video_path: str, frame_dir: str, fps=30, log=True):

    first_img = True
    writer = imageio.get_writer(video_path, fps=fps)

    file_list = sorted(os.listdir(frame_dir))
    for file_name in file_list:
        if not (file_name.endswith('jpg') or file_name.endswith('png')):
            continue

        fn = os.path.join(frame_dir, file_name)
        curImg = imageio.imread(fn)

        if first_img:
            H, W = curImg.shape[0:2]
            if log:
                print('img shape', (H, W))
            first_img = False

        writer.append_data(curImg)

    writer.close()


def extract_audio(input_file):
    video = VideoFileClip(input_file)
    audio = video.audio
    return audio


def add_audio(input_video, audio_clip, output_file):
    video = VideoFileClip(input_video)
    video = video.set_audio(audio_clip)
    video.write_videofile(output_file, codec="libx264", audio_codec="aac")


if __name__ == "__main__":
    filename = "./data/demo_input.mp4"
    # video_down_sample(filename, down_factor=2)

    # l, r, t, b = [576, 576, 56, 0]
    # l, r, t, b = [608, 608, 56, 320]
    l, r, t, b = [608, 608, 56, 0]
    # new_w, new_h = 512, 512
    new_w, new_h = None, None

    # 1. Extract frames from video
    # extract_frames(
    #   filename,
    #   output_folder="./data/frames",
    #   crop_func=lambda frame: crop_func(frame, l, r, t, b, new_w, new_h),
    # )

    # 2. Run Zoe segmentation on the frames
    input_folder = "./data/frames"
    # run_zoe_segment(input_folder, output_folder=input_folder + "_zoe", skip_existing=False)
    input_img = os.path.join(input_folder, "0001.png")
    zoe_mask_bool = run_zoe_segment(
        input_img, output_folder=input_folder+"_zoe", skip_existing=False)
    zoe_mask_bool = zoe_mask_bool[0]
    # frame = cv2.imread(input_img)
    # frame[~zoe_mask_bool[:, :, np.newaxis].repeat(3, axis=2)] = 0

    # Detect face in the frame
    face_output_path = "face_test.png"
    x, y, w, h = face_detect(input_img, border=0, output_path=face_output_path)
    exit(-1)

    """
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

    # only keep mask below the face
    zoe_mask_bool[:y+h, :] = False
    mask_bool_to_image(zoe_mask_bool, os.path.join(
        input_folder, "0001_mask_face.png"))
