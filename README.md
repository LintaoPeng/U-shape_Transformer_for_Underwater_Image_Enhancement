# U-shape Transformer 

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2111.11843)
![issues](https://img.shields.io/github/issues/LintaoPeng/U-shape_Transformer)
![forks](https://img.shields.io/github/forks/LintaoPeng/U-shape_Transformer)
![stars](https://img.shields.io/github/stars/LintaoPeng/U-shape_Transformer)
![license](https://img.shields.io/github/license/LintaoPeng/U-shape_Transformer)

This repository is the official PyTorch implementation of U-shape Transformer  for Underwater Image Enhancement. ([arxiv](https://arxiv.org/abs/2111.11843), [Dataset](https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html), [pretrained models](https://github.com/JingyunLiang/SwinIR/releases), [visual results](https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html)). U-shape Transformer achieves **state-of-the-art performance** in underwater image enhancement task.

</br>



:rocket:  :rocket:  :rocket: **News**:

- 2021/11/24  We release the official code of U-shape Transformer

- 2021/11/23  We release LSUI dataset, We released a large-scale underwater image (LSUI) dataset including 5004 image pairs, which involve richer underwater scenes (lighting conditions, water types and target categories) and better visual quality reference images than the existing ones. You can download it from [[here\]](https://pan.baidu.com/s/1rtHIwEmVp9BZDYJ_kb5Wfg). The password is 1iya.

  ![avatar](./figs/data.png)

---

> The light absorption and scattering of underwater impurities lead to poor underwater imaging quality. The existing data-driven based underwater image enhancement (UIE) techniques suffer from the lack of a large-scale dataset containing various underwater scenes and high-fidelity reference images. Besides, the inconsistent attenuation in different color channels and space areas is not fully considered for boosted enhancement. In this work, we constructed a large-scale underwater image (LSUI) dataset including 5004 image pairs, and reported an U-shape Transformer network where the transformer model is for the first time introduced to the UIE task. The U-shape Transformer is integrated with a channel-wise multi-scale feature fusion transformer (CMSFFT) module and a spatial-wise global feature modeling transformer (SGFMT) module, which reinforce the network's attention to the color channels and space areas with more serious attenuation. Meanwhile, in order to further improve the contrast and saturation, a novel loss function combining RGB, LAB and LCH color spaces is designed following the human vision principle. The extensive experiments on available datasets validate the state-of-the-art performance of the reported technique with more than 2dB superiority.
> 
><p align="center">
  <img width="800" src="./figs/u_shape_trans.png">
</p>

#### Contents

1. [Training](#Training)
1. [Testing](#Testing)
1. [Results](#Results)
1. [Citation](#Citation)
1. [License and Acknowledgement](#License-and-Acknowledgement)


### Training

If you need to train our U-shape transformer from scratch, you need to download our lsui dataset from [LSUI](https://pan.baidu.com/s/1rtHIwEmVp9BZDYJ_kb5Wfg) with the password 1iya, and then randomly select 4500 picture pairs as the training set to replace the data folder, and the remaining 504 as the test set to replace the test folder.

Then, run the train.ipynb file with Jupiter notebook, and the trained model weight file will be automatically saved in saved_ Models folder. As described in the paper, we recommend you use L2 loss for the first 600 epochs and L1 loss for the last 200 epochs.

Environmental requirements:

- Python 3.7 or a newer version

- Pytorch 1.7 0r a newer version

- CUDA 10.1 or a newer version

- OpenCV 4.5.3 or a newer version

- Jupyter Notebook

