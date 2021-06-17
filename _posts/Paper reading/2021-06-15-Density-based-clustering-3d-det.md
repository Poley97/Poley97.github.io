---
layout: post
title: 'Density Based Clustering for 3D Object Detection in Point Clouds'
date: 2021-06-15
author: Poley
cover: '/assets/img/20210615/DBC.png'
tags: 论文阅读
---

> 论文链接： https://openaccess.thecvf.com/content_CVPR_2020/papers/Ahmed_Density-Based_Clustering_for_3D_Object_Detection_in_Point_Clouds_CVPR_2020_paper.pdf

# Introduction

处理不规则点云数据的瓶颈在于处理点的数量。点的数量直接影响了模型的大小和复杂度，以及输出特征的质量。因此本文使用增量的减少点云的数量来解决这个问题。

提出了一个新的pipeline，使用三个独立的模块来完成相应的任务，分别是
1. background-foreground segmentation, 使用PointNet++, 通过前景背景分割来去除容易产生FP的背景点，也自然的降低了点云的数量； 
2. class agnostic instance segmentation，使用无监督的DBSCAN，之后预测一个和目标中心的偏移（类似于vote）；
3. 3D object detection, 使用Edge-aware PointNet，预测目标的具体信息。

![](/assets/img/20210615/DBCF1.png)

# Network Architecture

使用这种级联模式的主要原因是为了减小除了点的数量，SOAT的一些方法一般处理1024-5000个点。但是如果没有分割网络，随机采样的点只会有一小部分对于最终的特征有贡献。在本文提出的方法中，通过前景背景分割来剔除背景点，即显著降低了点的数量，同时保留了有意义的点。之后的模块也会保证有意义的点被保留，得到更focused感受野。

## Background/Foreground Segmentation
使用PointNet++来做前景背景预测，但是和原版不同的是，一般的分割都在全部点云上做，这里为了提高速度，只选取了其中的N个点来做分割。但是这样网络就只能得到这N个点的语义标签，为了得到全部点的语义标签，**本文提出
Nearest Neighbor - Upsampling (NN-Ups），使用近邻中语义标签置信度最高的作为上采样点的语义标签**。

$$
\begin{equation}
S\left(\hat{y}_{p}\right)=\arg \max \left\{S\left(y_{1}\right) \ldots S\left(y_{k}\right)\right\}
\end{equation}
$$

## Class Agnostic Instance Segmentation
传统的vote是对每一个前景点都预测一个和其对应的目标中心的偏移，但是这样的效果并不是很好，因为预测值的方差太大，不好收敛。因此这里提出了一个新的思路。先使用无监督的聚类方法DBSCAN来对前景点进行聚类，聚类之后得到聚类的中心，**去预测这个聚类中心和其对应的目标中心的偏移**。

![](/assets/img/20210615/DBCF2.png)

DBSCAN的示意图如下所示，属于聚类的基本方法，不介绍了。
![](/assets/img/20210615/DBCF3.png)

Loss如下，预测每个点对应的聚类的中心到目标中心的偏移。
$$
L_{i n s}=\frac{1}{N} \sum_{i}^{N}\left\|\triangle C_{i}-\triangle C_{i}^{*}\right\|
$$
where $\triangle C_{i}$ is the true offset between $C_{i_{D B}}$ and $C_{i}$, while $\triangle C_{i}^{*}$ is the predicted offset vector for every point $i$ in the point cloud.

这样做有两个好处

1. **引入几何信息**：因为聚类引入了几何信息，因此可以保证在进一步下采样的时候保留每个聚类的信息，而FPS或者RS都可能会丢失某些目标的信息

2. **加快收敛**： 由于预测的是聚类中心到目标中心的距离，这个偏移值一般更小，所以网络更容易收敛，并且得到的结果也更加准确。同样，聚类的归属也通过之前的上采样方法拓展到全部的点云信息。

## EPN: EdgeAware PointNet

一个PointNet++的升级版，用来回归最后的偏差。

## Bayesian Uncertainty Estimation

*Dropout as a bayesian approximation: Representing model uncertainty in deep learning*提出使用Dropout来估计深度学习的不确定度，本文同样在PointNet++中添加了至少一层dropout层，用来计算感受野的方差。这个方差用于滤除和不确定度有关系的点，来提高预测精度。

![](/assets/img/20210615/DBCF4.png)