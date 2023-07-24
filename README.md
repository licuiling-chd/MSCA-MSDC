# Computationally Lightweight Hyperspectral Image Classification Using a Multiscale Depthwise Convolutional Network With Channel Attention

This repository is the implementation of our paper: [Computationally Lightweight Hyperspectral Image Classification Using a Multiscale Depthwise Convolutional Network With Channel Attention](https://ieeexplore.ieee.org/document/10150418). 

If you find this work helpful, please cite our paper:

    @ARTICLE{10150418,
    author={Ye, Zhen and Li, Cuiling and Liu, Qingxin and Bai, Lin and Fowler, James E.},
    journal={IEEE Geoscience and Remote Sensing Letters}, 
    title={Computationally Lightweight Hyperspectral Image Classification Using a Multiscale Depthwise Convolutional Network With Channel Attention}, 
    year={2023},
    volume={20},
    number={},
    pages={1-5},
    doi={10.1109/LGRS.2023.3285208}}
 
 ## Descriptions
 
Convolutional networks have been widely used for the classification of hyperspectral images; however, such networks are notorious for their large number of trainable parameters and high computational complexity. Additionally, traditional convolution-based methods are typically implemented as a simple cascade of a number of convolutions using a single-scale convolution kernel. In contrast, a lightweight multiscale convolutional network is proposed, capitalizing on feature extraction at multiple scales in parallel branches followed by feature fusion. In this approach, 2-D depthwise convolution is used instead of conventional convolution to reduce network complexity without sacrificing classification accuracy. Furthermore, multiscale channel attention (MSCA) is also employed to selectively exploit discriminative capability across various channels. To do so, multiple 1-D convolutions with varying kernel sizes provide channel attention at multiple scales, again with the goal of minimizing network complexity. Experimental results reveal that the proposed network not only outperforms other competing lightweight classifiers in terms of classification accuracy, but also exhibits a lower number of parameters as well as significantly less computational cost.

