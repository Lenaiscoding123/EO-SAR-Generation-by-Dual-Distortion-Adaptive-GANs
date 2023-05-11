# [Remote Sensing] Two-Way Generation of High-Resolution EO and SAR Images via Dual Distortion-Adaptive GANs
This repository contains the official implementation of paper [Two-Way Generation of High-Resolution EO and SAR Images via Dual Distortion-Adaptive GANs](https://www.mdpi.com/2072-4292/15/7/1878). The dataset "SN6-SAROPT" proposed in the paper can be downloaded through the following link:

- [Google Drive](https://drive.google.com/drive/folders/1lAhwnO-nePM2X61MEB0eazdNlMmYJegi?usp=share_link) 

## Introduction
To tackle the non-linear geometric distortions between EO and SAR images, we propose a novel image translation algorithm designed to handle such distortions adaptively and introduce a new dataset SN6-SAROPT with sub-meter resolution to feature the challenges of modality transfer tasks in fine-scale remote sensing imagery.

For more details, please refer to our [orginal paper](https://www.mdpi.com/2072-4292/15/7/1878).


## Requirement
* Python        (3.6)
* torch         (1.8.0+cu111)
* visdom        (0.1.8.9)
* skimage       (0.15.0)
* Yaml          (5.4.1)
* cv2           (3.4.2)


## Usage
### Training
If you want to train the model
1. `git clone git@github.com:Lenaiscoding123/EO-SAR-Generation-by-Dual-Distortion-Adaptive-GANs.git`
2. download the training data [SN6-SAROPT ](https://drive.google.com/drive/folders/1lAhwnO-nePM2X61MEB0eazdNlMmYJegi?usp=share_link) or other EO-SAR datasets (e.g., SEN1-2)
3. unzip and put the downloaded images to the corresponding path:
```
data
├── dataset1_name
│     ├── TRAIN
│     │    ├── A
│     │    └── B
│     └── VAL
│          ├── A
│          └── B
├── dataset2_name
.     ├── TRAIN
.     │    ├── A
.     │    └── B
.     └── VAL
.          ├── A
.          └── B
```

4. Start visdom：
 ```
python -m visdom.server -p 6019
```

5. Train:
 ```
python train.py
```

### Testing
 ```
python test.py
```

## References

The other two datasets used in the paper are obtained from:
- [SAR2Opt](https://github.com/MarsZhaoYT/SAR2Opt-Heterogeneous-Dataset)
- [SEN1-2](https://mediatum.ub.tum.de/1436631)



Our implementation is based on [Reg-GAN](https://github.com/Kid-Liet/Reg-GAN). We would like to thank them.

Citation
-----

In case of use, please cite our publication:

Qing, Yuanyuan, et al. "Two-Way Generation of High-Resolution EO and SAR Images via Dual Distortion-Adaptive GANs." Remote Sensing 15.7 (2023): 1878.

Bibtex:
```

@Article{rs15071878,
AUTHOR = {Qing, Yuanyuan and Zhu, Jiang and Feng, Hongchuan and Liu, Weixian and Wen, Bihan},
TITLE = {Two-Way Generation of High-Resolution EO and SAR Images via Dual Distortion-Adaptive GANs},
JOURNAL = {Remote Sensing},
VOLUME = {15},
YEAR = {2023},
NUMBER = {7},
ARTICLE-NUMBER = {1878},
URL = {https://www.mdpi.com/2072-4292/15/7/1878},
ISSN = {2072-4292},
DOI = {10.3390/rs15071878}
}

```




