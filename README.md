# Social Motion Prediction with Cognitive Hierarchies (NeurIPS 2023)

<p align="center">
<a href="https://arxiv.org/pdf/2311.04726.pdf">
<img src="https://img.shields.io/badge/arXiv-2311.04726-b31b1b.svg?style=for-the-badge"></a>
<a href="https://walter0807.github.io/Social-CH/">
<img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=Google%20chrome&logoColor=white" alt="Project Page"></a>
<a href="https://www.youtube.com/watch?v=pVBICYpGhyU&ab_channel=JasonQin">
<img src="https://img.shields.io/badge/YouTube-video-black?style=for-the-badge&logo=youtube&logoColor=white&labelColor=FF0000&color=black" alt="YouTube video"></a>
</p>

This is the official PyTorch implementation of the paper "Social Motion Prediction with Cognitive Hierarchies" (NeurIPS 2023).

![](https://walter0807.github.io/Social-CH/assets/teaser.gif)

## Dependencies
- python 3.9
- pytorch 1.13.1
- [torch_dct](https://github.com/zh217/torch-dct)

## Dataset
Please refer to [Wusi Basketball Training Dataset](data/).

## Train
```bash
python train.py \
--config configs/wusi_ch.yaml \
--train
```

## Test
```bash
python train.py \
--config configs/wusi_ch.yaml \
--eval --ckpt checkpoint/wusi_ch/best.pth
```

You can download the pretrained model weight from [here](https://drive.google.com/drive/folders/1qp2t5lXq2J2pdV6TxLLXwzITfTvltzyD?usp=sharing).

## Citation
If you find our work useful for your project, please cite the paper:
```bibtex
@inproceedings{zhu2023social,
    title={Social Motion Prediction with Cognitive Hierarchies},
    author={Zhu, Wentao and Qin, Jason and Lou, Yuke and Ye, Hang and Ma, Xiaoxuan and Ci, Hai and Wang, Yizhou},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
}
```

## Acknowledgement
This repo is built on [MRT](https://github.com/jiashunwang/MRT). The motion capture algorithm is based on [Faster-VoxelPose](https://github.com/AlvinYH/Faster-VoxelPose). Thank the authors for releasing their codes. 