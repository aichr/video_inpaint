# Video Editing

## Intro
This is a simple pipeline to edit the demo video by inpainting. For the Heygen sample video, the pipeline works as follows

- Extract frames and run get the `person mask`, by running a depth estimator ([ZoeDepth](https://github.com/isl-org/ZoeDepth)) and then seperate the foreground from backgroud.

- Run face detector on each frame to get the `face mask`. The difference of `person mask` minus `face mask` is the region that needs inpainting.

- Run stable diffusion inpainting (with controlnet) on a bunch of fashion prompts to fill the new clothes for the person.

- (Manual) Pick the good generated key images.

- Use ebsynth to synthesize the frames, and concat the frames into the final mp4.



https://github.com/aichr/video_inpaint/assets/113976014/f34ea1be-1b28-465e-a7e0-56c3e6d9255b


Example stable diffusion inpainting results for keyframes:

<p float="left">
  <img src="https://github.com/aichr/video_inpaint/assets/113976014/ff568670-b865-444a-b36a-b4e88cedb606" width="200" />
  <img src="https://github.com/aichr/video_inpaint/assets/113976014/70937ced-6f39-459e-97f8-ec9ace7a4d36" width="200" /> 
  <img src="https://github.com/aichr/video_inpaint/assets/113976014/952b5931-ff87-4e31-9799-a36019578a5c" width="200" />
  <img src="https://github.com/aichr/video_inpaint/assets/113976014/06681d6d-a51f-4b43-9ae5-e481bc820ef4" width="200" />
</p>

## Installation
```
pip install -r requirements.txt
```

### Dependency
```
# zoe depth
git clone https://github.com/isl-org/ZoeDepth

# ebsynth
git clone https://github.com/SingleZombie/ebsynth.git
python install.py

# optional: face parsing, for getting better face masks for inpainting
git clone https://github.com/zllrunning/face-parsing.PyTorch.git
python gdown.py
```

## Usage

```
# prepare data
python prepare_data.py

# run the pipeline (-ri: inpainting, -re: ebsynth, -rf: ffmpeg)
python pipeline.py -ri -re -rf

```

# Limitations

- The example output is generated by the [ebsynth app](https://ebsynth.com/) on Mac.
The programmable version of `ebsynth` used in this repo doesn't seem to support the advanced features such as de-clicker, which leads to inferior results if
you directly run the `pipeline.py`. See this [upper body](results/output_upper_body.mp4) video (with audio) as the raw output of the pipeline.


https://github.com/aichr/video_inpaint/assets/113976014/386c8676-5fac-4453-adb0-5d993b997969


- The identity-preserving feature is naturally achieved by fixing the face unchanged using inpainting masks, so we can't edit the faces yet. To have more editing flexibility, 
I tried [Instruct-Pix2Pix](inst_pix2pix.py) to achieve wilder editing but the output quality isn't consistent.

- Having issue generating good hands for [Half body](results/output_half_body.mp4).



https://github.com/aichr/video_inpaint/assets/113976014/d713bd2a-e027-4e4e-a661-1e62512e966f


