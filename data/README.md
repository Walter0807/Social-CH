# Wusi Basketball Training Dataset

![](https://walter0807.github.io/Social-CH/assets/wusi_demo.jpg)

## Description
The Wusi Basketball Training Dataset is proposed in *Social Motion Prediction with Cognitive Hierarchies (NeurIPS 2023)*. It is a multi-person 3D motion dataset with a special focus on strategic interactions. This new dataset contains 3D skeleton representation of 5 people in highly intense basketball drills recorded at 25 FPS.

Please check our [paper](https://openreview.net/pdf?id=lRu0dN7BY6) and the [project webpage](https://neurips.cc/virtual/2023/poster/70595) for more details. 


## Dependencies

Requirements:

- Numpy

## Download

[Google Drive](https://drive.google.com/drive/folders/1UGp1ejyVXZ-fjPqyKfbyL-5r-xCzkZr0)

Please read carefully the [license agreement](LICENSE.md) before you download and/or use the Wusi dataset. By downloading and/or using the dataset, you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them.

1. We provide the processed training and testing split in numpy (sequence, people, frame, keypoints).
   - `training.npy`: (7125, 5, 50, 45)
   - `training.npy`: (1782, 5, 50, 45)

2. If you wish to handle the full motion sequences and process them on your own, please download the `Wusi` folder to `data/` and check the processing guidance below. 

3. If you wish to fit the 3D poses to SMPL for better visual effect, please refer to this [repo](https://github.com/AlvinYH/joint2smpl).

## Usage
The script (`process_wusi.py`) reads the undivided data and cut them up into sequences, you can specify the sequence length by argument `sequence_len` and stride by the argument `stride`. After processing the undivided data, it shall tell you the sequence number and sequence length. Then, processed data will be divided into training and testing set. You could specify the ratio of training dataset by argument `ratio`, the remaining should automatically become test set.

```bash
cd data
python precess_wusi.py --stride=[your stride] --sequence_len=[your sequence len] --ratio=[your ratio]
```

## Citation

If you use this dataset, please cite the corresponding NeurIPS 2023 paper:
```bibtex
@inproceedings{zhu2023social,
    title={Social Motion Prediction with Cognitive Hierarchies},
    author={Zhu, Wentao and Qin, Jason and Lou, Yuke and Ye, Hang and Ma, Xiaoxuan and Ci, Hai and Wang, Yizhou},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
}
```