# Contrastive Learning with Non-Semantic Negatives

This repository is the official implementation of [Robust Contrastive Learning Using Negative Samples with Diminished Semantics](https://arxiv.org/). Contrastive learning utilizes positive pairs which preserve semantic information while perturbing superficial features in the training images. Similarly, we propose to generate negative samples to make the model more robust, where only the superfluous instead of the semantic features are preserved. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/22885450/137439195-4ebf806f-23bb-43d3-9134-09a8a939a6a0.png" width="800">
</p>

## Preparation

Install PyTorch and check `preprocess/` for ImageNet-100 and ImageNet-Texture preprocessing scripts.

Install the texture-synthesis tool from https://github.com/EmbarkStudios/texture-synthesis
Run `python preprocess/imangenet_texture_syn.py`

## Training

The following code is used to pre-train MoCo-v2 + patch / texture-based NS. The major code is developed with minimal modifications from the [official implementation](https://github.com/facebookresearch/moco). 

```train
python moco-non-sem-neg.py -a resnet50 --lr 0.03 --batch-size 128 --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos --moco-k 16384 \
  --robust nonsem --num-nonsem 1 --alpha 2 --epochs 200 --patch-ratio 16 72 \
  --ckpt_path ./ckpts/mocov2_mocok16384_bs128_lr0.03_nonsem_16_72_noaug_nn1_alpha2_epoch200  \
  /path/to/imagenet-100/ 

python moco-non-sem-neg.py -a resnet50 --lr 0.03 --batch-size 128 --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos --moco-k 16384 \
  --robust texture_syn --num-nonsem 1 --alpha 2 --epochs 200 \
  --ckpt_path ./ckpts/mocov2_mocok16384_bs128_lr0.03_texture_nn1_alpha2_epoch200 \
  /path/to/imagenet-100-texture/ 
```

* Change `/path/to/imagenet-100/` with the ImageNet-100 dataset directory. 
* Change `--alpha` and `-moco-k` to reproduce results with different configurations.

## Linear Evaluation

Run following code is used to reproduce MoCo-v2 + patch-based NS model reported in Table 1. 

```eval
python main_lincls.py -a resnet50 --lr 10.0 --batch-size 128 --epochs 60 \
  --pretrained ./ckpts/mocov2_mocok16384_bs128_lr0.03_nonsem_16_72_noaug_nn1_alpha2_epoch200/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --ckpt_path ./ckpts/mocov2_mocok16384_bs128_lr0.03_nonsem_16_72_noaug_nn1_alpha2_epoch200 \
  /path/to/imagenet-100/ 
```

## Pre-trained Models

You can download pretrained models here:

|                              | ImageNet         | ImageNet-C       | ImageNet-S       | Stylized-ImageNet | ImageNet-R       |
|------------------------------|------------------|------------------|------------------|-------------------|------------------|
| MoCo-v2 - k=16384            | 77.88 ± 0.28     | 43.08 ± 0.27     | 28.24 ± 0.58     | 16.20 ± 0.55      | 32.92 ± 0.12     |
| + Texture-based - alpha=2    | 77.76 ± 0.17     | 43.58 ± 0.33     | 29.11 ± 0.39     | 16.59 ± 0.17      | 33.36 ± 0.15     |
| + Patch-based - alpha=2      | **79.35** ± 0.12 | **45.13** ± 0.35 | 31.76 ± 0.88     | 17.37 ± 0.19      | 34.78 ± 0.15     |
| + Patch-based - alpha=3      | 75.58 ± 0.52     | 44.45 ± 0.15     | **34.03** ± 0.58 | **18.60** ± 0.26  | **36.89** ± 0.11 |
| MoCo-v2 - k=8192             | 77.73 ± 0.38     | 43.22 ± 0.39     | 28.45 ± 0.36     | 16.83 ± 0.12      | 33.19 ± 0.44     |
| + Patch-based - alpha=2      | **79.54** ± 0.32 | **45.48** ± 0.20 | **33.36** ± 0.45 | **17.81** ± 0.32  | **36.31** ± 0.37 |

