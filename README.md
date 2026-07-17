[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

## Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning

**\*\*\*\*\*\*\*\*\* June 19, 2020 \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*** <br>
The code and data are going through the internal review and will be released later!

**\*\*\*\*\*\*\*\*\* August 26, 2020 \*\*\*\*\*\*\*\*\*\*\*\*** <br>
The dataset is still going through the internal review, please wait.

**\*\*\*\*\*\*\*\*\* September 7, 2020 \*\*\*\*\*\*\*\*\*\*** <br>
The code & pose data are released!

**\*\*\*\*\*\*\*\*\* March 20, 2022 \*\*\*\*\*\*\*\*\*\*\*\*\*** <br>
Two versions of codebase are released.

### Summary

- [Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning](#Dance-Revolution-Long-Term-Dance-Generation-with-Music-via-Curriculum-Learning)
  - [Summary](#summary)
  - [Introduction](#introduction)
  - [Project](#project)
  - [Requirements](#requirements)
  - [Dataset and Installation](#dataset-and-installation)
  - [Pre-trained Model](#pre-trained-model)
  - [Training Issues](#training-issues)
  - [Inference](#inference)
  - [Future Work](#future-work)
  - [Acknowledgment](#acknowledgment)

### Introduction

This repo is the PyTorch implementation of "[Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning](https://stonyhu.github.io/dancerev/)". Our proposed approach significantly outperforms the existing state-of-the-arts (SOTAs) in extensive experiments, including automatic metrics and human judgements. It can generate creative long dance sequences, e.g., about <strong>1-minute length under 15 FPS</strong>, from the input music clips, which are natural-looking, diverse and style-consistent with the music from test set. With the help of 3D human pose estimation and 3D animation driving, this technique can be used to drive various 3D character models such as the 3D model of Hatsune Miku (very popular virtual character in Japan), and has the great potential for the virtual advertisement video generation.

### Project

Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning. arXiv:2006.06119, 2020. <br/>
Ruozi Huang*, Huang Hu*, Wei Wu, Kei Sawada, Mi Zhang and Daxin Jiang. <br/>
[[Project]](https://stonyhu.github.io/dancerev/) [[Paper]](https://openreview.net/pdf?id=xGZG2kS5bFk) [[Video]](https://pan.baidu.com/s/1rKFtrGWBG8BEjPQmDkJbJg?pwd=wnnn)

### Requirements

- Python 3.7
- PyTorch 1.6.0
- ffmpeg

### Dataset and Installation

- We released the dance pose data and the corresponding audio data into [[here]](https://pan.baidu.com/s/1_5GU9efw1PaOwnUKulQ-Kw?pwd=dgxv).
  Please put the downloaded `data/` into the project directory `DanceRevolution/` and run `prepro.py` that will generate the training data directory `data/train_1min` and test data directory `data/test_1min`. The tree directory of `data/` is displayed as follow:

```
data/
├── audio/
|   ├── ballet_1min/
|   ├── hiphop_1min/
|   ├── pop_1min/
├── json/
|   ├── ballet_1min/
|   ├── hiphop_1min/
|   ├── pop_1min/
├── train_1min/
├── test_1min/
```

- Dirs
  - `audio`: the directory to store .m4a file for three styles respectively
  - `json`: the directory to store dance motion sequences for three styles respectively
  - `train_1min`: the training dance motion sequences (1 min)
  - `test_1min`: the test dance motion sequences (1 min)

The default `data/train_1min` and `data/test_1min` is our train/test division. The pose sequences are extracted from the collected dance videos with original 30FPS while the audio data is m4a format. After the generation is finished, you can run `interpolate_to30fps.py` to increase the 15FPS to 30FPS to produce visually smoother dance videos for the generated results.

- If you plan to train the model with your own dance data (2D), please install [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for the human pose extraction. After that, please follow the hierarchical structure of directory `data/` to place your own extracted data and run `prepro.py` to generate the training data and test data. Note that, we develope a `interpolate_missing_keyjoints.py` script to find the missing keyjoints to reduce the noise in the pose data, which is introduced by the imperfect extraction of OpenPose.

- Due to the lack of 3D dance data in hand, we did not test our approach on 3D data. Recently, there is a wonderful paper conducted by Li et al., "[AI Choreographer: Music Conditioned 3D Dance Generation with AIST++](https://arxiv.org/abs/2101.08779)", on music-conditioned 3D dance generation, which releases a large-scale 3D human dance motion dataset, [AIST++](https://google.github.io/aistplusplus_dataset/factsfigures.html). Just heard from the authors of this paper that our approach also works on their released dataset.

### Pre-trained Model

We train the model according to the following command using the v2 code:

```python
python3 train.py --train_dir ../data/train_1min --test_dir ../data/test_1min \
                 --output_dir checkpoints/layers1_win900_schedule100_condition10_detach \
                 --batch_size 32 --seq_len 900 --max_seq_len 4500 \
                 --d_frame_vec 438 --d_pose_vec 50 --frame_emb_size 200 --pose_emb_size 50 \
                 --n_layers 1 --n_head 8 --d_inner 1024 --d_k 64 --d_v 64 \
                 --sliding_windown_size 900 --condition_step 10 --lambda_v 0.01 \
                 --lr 0.0001 --epoch 10000
```

And release the checkpoint at 5500-th epoch as follows (continue to update) and you can run the `v2/test_1min.sh` to load the model and generate dance samples using the music clips from the test set:

|                                          |                 n_heads=8,d_inner=1024,d_model=200,d_k=d_v=64                 |
| ---------------------------------------- | :---------------------------------------------------------------------------: |
| **n_layers=1,seq_len=900,lambda_v=0.01** | [[Model_epoch5500]](https://pan.baidu.com/s/1AhqdH5pQfiMN9kIRYNpiqg?pwd=fr62) |

### Training Issues

We released two versions of codebases that have passed the test. In V1 version, the local self-attention module is implemented base on the [longformer](https://github.com/allenai/longformer) that provides the custom CUDA kernel to accelerate the training speed and save GPU memory for long sequence inputs. While V2 version just implements the local self-attention module via the naive PyTorch implementation, i.e., the attention mask operations. In practice, we found the performance of V2 is more stable and recommend to use V2 version. Here are some training tricks that may be helpful for you:

- Small batch sizes, such as 32 and 48, would help model to converge better and the model usually converges well at around the 3000-th epoch. While considering the different environments, we suggest to try different hyperparameters and different epochs.
- Increasing sliding window size of local self-attention is beneficial to the more stable performance while the cost (e.g., training time and GPU memory usage) would become high. This point has been empirically justified in the ablation study of encoder structures in the paper. So if you are free of GPU resource limitation, we recommend to use the large sliding window size for training.

### Inference

The inference pipeline has been released, you can run the following command to generate dance motion sequences for the test music:

```python
python3 inference.py --test_dir music/demo_song \
                     --output_dir outputs/demo_song \
                     --model checkpoints/epoch_best.pt \
                     --dance_num 6
```

- Args
  - `test_dir`: the directory to store input .m4a file. Recommend to split the whole music into 1-min clips.
  - `output_dir`: the directory to store generated dance motion sequences
  - `model`: the best checkpoint
  - `dance_num`: the number of dance motion sequences needs to be generated for one song. Please set it to an even number.

Moreover, we also provide the `inference.sh` to generate dance motion sequences (15FPS) for the test music. Then we increase the FPS of generated motion sequences from 15FPS to 30FPS by the linear interpolation and synthesize final dance videos for the generated results. Note that, our system currently only supports the `m4a` music format. While you can use [Online Audio Converter](https://online-audio-converter.com/) to convert other music format into `m4a` with **Standard 128k Quality option**.

### Future Work

There are some possible future works worth the further exploration:

- More style and rhythm consistent methods for the alignment of the input music and the output dance motion sequences.
- End-to-end methods that could work with the raw audio wave rather than the pre-extracted music features.
- Large-scale Seq2Seq Pre-trained Models for the music-to-dance synthesis, such as the potential DanceGPT.
- Better automatic evaluation methods for assessing the quality of generated dance motion sequences. Since Fréchet inception distance (FID) is the standard metric for evaluating the quality of generated images by GAN. In practice, we found this metric is not suitable for assessing the quality of dance motion sequences.
- More precise detection methods for the dance motion beats.

### Generated Example Videos

- Ballet style
<p align='center'>
  <img src='imgs/ballet-1.gif' width='400'/>
  <img src='imgs/ballet-2.gif' width='400'/>
</p>

- Hiphop style
<p align='center'>
  <img src='imgs/hiphop-1.gif' width='400'/>
  <img src='imgs/hiphop-2.gif' width='400'/>
</p>

- Japanese Pop style
<p align='center'>
  <img src='imgs/pop-1.gif' width='400'/>
  <img src='imgs/pop-2.gif' width='400'/>
</p>

- Photo-Realisitc Videos by [Video-to-Video](https://github.com/NVIDIA/vid2vid) </br>
We map the generated skeleton dances to the photo-realistic videos by [Video-to-Video](https://github.com/NVIDIA/vid2vid). Specifically, We record a random dance video of a team memebr to train the vid2vid model. Then we generate photo-realistic videos by feeding the generated skeleton dances to the trained vid2vid model. Note that, our team member has authorized us the usage of her portrait in following demos.
<p align='center'>
  <img src='imgs/skeleton-1.gif' width='428'/>
  <img src='imgs/kazuna-crop-1.gif' width='400'/>
  <img src='imgs/skeleton-2.gif' width='429'/>
  <img src='imgs/kazuna-crop-2.gif' width='400'/>
</p>

- Driving 3D model by applying [3D human pose estimation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ci_Optimizing_Network_Structure_for_3D_Human_Pose_Estimation_ICCV_2019_paper.pdf) and Unity animation to generated skeleton dances.

### Acknowledgment

Special thanks to the authors of [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), [LongFormer](https://github.com/allenai/longformer), [Video-to-Video](https://github.com/NVIDIA/vid2vid), [3D Human Pose Reconstruction](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ci_Optimizing_Network_Structure_for_3D_Human_Pose_Estimation_ICCV_2019_paper.pdf) and [Dancing2Music](https://github.com/NVlabs/Dancing2Music).
