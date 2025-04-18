# Efficient Video Compression via Content-Adaptive Super-Resolution
We present a new approach that augments existing codecs with a small, content-adaptive super-resolution model that significantly boosts video quality. Our method, SRVC, encodes video into two bitstreams: 
1. a `content stream`, produced by compressing downsampled low-resolution video with the existing codec, 
2. a `model stream`, which encodes periodic updates to a lightweight super-resolution neural network customized for short segments of the video.

SRVC decodes the video by passing the decompressed low-resolution video frames through the (time-varying) super-resolution model to reconstruct high-resolution video frames.
## Installation
For installing the required packages using [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), use:
```
git clone https://github.com/AdaptiveVC/SRVC.git
cd SRVC
conda env create -f environment.yml
```

## Running SRVC
For running SRVC, check `python srvc.py --help`.
python srvc.py /data/VSR_Project/srvc/SRVC/datasets/yuv/test1.yuv /data/VSR_Project/srvc/SRVC/datasets/resize/test1_480x270.yuv /data/VSR_Project/srvc/SRVC/output_new "" "srvc" 0.0005 16 0 100 1 5 0.1 5 "24" 0 False False True True True "1920,1080" "480,270" False
## Videos
### Vimeo Short Films
We use 28 short films with direct download buttons on Vimeo. These videos are in high-resolution, have realistic scene changes from movie-makers, and have 10min+ duration. 

Vimeo `video_id`s are located [here](./datasets/vimeo/vimeo_ids.txt), and you can view/download them at `vimeo.com/video_id`, e.g., [vimeo.com/441417334](https://vimeo.com/441417334).

Once downloaded, you can use the scripts and instructions located [here](./video_processing_scripts/README.md) to pre-process the data into the low-resolution and high-resolution formats needed by SRVC.

### Xiph Full Sequences
We use the following four long video sequences from the [Xiph](https://media.xiph.org/video/derf/) video dataset:
- [Big Buck Bunny](https://media.xiph.org/video/derf/y4m/big_buck_bunny_1080p24.y4m.xz)
- [Elephants Dream](https://media.xiph.org/video/derf/y4m/elephants_dream_1080p24.y4m.xz)
- [Sita Sings the Blues](https://media.xiph.org/video/derf/y4m/sita_sings_the_blues_1080p24.y4m.xz)
- [Meridian](https://media.xiph.org/video/derf/meridian/MERIDIAN_SHR_C_EN-XX_US-NR_51_LTRT_UHD_20160909_OV/)


## Citation
You may cite this work using:
```
@InProceedings{Khani_2021_ICCV,
    author    = {Khani, Mehrdad and Sivaraman, Vibhaalakshmi and Alizadeh, Mohammad},
    title     = {Efficient Video Compression via Content-Adaptive Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4521-4530}
}
```
