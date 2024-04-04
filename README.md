# Video Editing

## Intro
This is a simple pipeline to edit the demo video by inpainting. For the Heygen sample video, the pipeline works as follows

- Extract frames and run get the `person mask`, by running a depth estimator ([ZoeDepth](https://github.com/isl-org/ZoeDepth)) and then seperate the foreground from backgroud.

- Run face detector on each frame to get the `face mask`. The difference of `person mask` minus `face mask` is the region that needs inpainting.

- Run stable diffusion inpainting (with controlnet) on a bunch of fashion prompts to fill the new clothes for the person.

- (Manual) Pick the good generated key images.

- Use ebsynth to synthesize the frames, and concat the frames into the final mp4.

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

- The programmable version of `ebsynth` used in this repo doesn't seem to support the advanced features such as de-clicker, which leads to inferior results if
you directly run the `pipeline.py`. See this [upper body](results/output_upper_body.mp4) video as the raw output of the pipeline.

- Current fix face unchanged by inpainting masks, so we can't edit the faces yet. To have more editing flexibility, 
I tried [Instruct-Pix2Pix](https://www.timothybrooks.com/instruct-pix2pix) to achieve wilder editing but the output quality isn't consistent.

- Having issue generating good hands for [Half body](results/output_half_body.mp4).