---
layout: post
title: 'Attentional PointNet for 3D-Object Detection in Point Clouds'
date: 2021-05-29
author: Poley
cover: '/assets/img/20210529/AP.png'
tags: 论文阅读
---

>论文链接 ： https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Paigwar_Attentional_PointNet_for_3D-Object_Detection_in_Point_Clouds_CVPRW_2019_paper.pdf


>参考博客 : 

所提出的网络主要包含以下部分：**Context Network, Recurrent Localization Network, 3D Transformer, and Resampler, Classifier, 3D Box Estimation.** 并且设计了一个独特的loss。

## Context Network
输入一个cropped region（$12m\times 12m$）的3D点。和一个点云的2D垂直投影，在$120\times 120$的cell里。使用简化版的PointNet，也就是没有T-Net的来提取特征。

![](/assets/img/20210529/APF3.png)

Height map就是一个2D数据，直接用标准卷积即可。

这两种信息是互补的，3D可以帮助区别2D中容易混淆的物体。2D可以更直观的分辨3D中不好辨识的物体。

## Recurrent Localization Network

使用一个**GRU**层来对特征进行处理，使用前面给出的特征以及上一次迭代的状态来产生此时刻的状态。然后估计一个刚性变换的5个参数。$\left(\cos \theta_{i}, \sin \theta_{i}, T x_{i}, T y_{i}, T z_{i}\right) \in \Theta_{i}$。得到一个刚性变换矩阵

$$
\begin{equation}
T\left(\Theta_{i}\right)=\left[\begin{array}{cccc}
\cos \theta_{i} & -\sin \theta_{i} & 0 & T x_{i} \\
\sin \theta_{i} & \cos \theta_{i} & 0 & T y_{i} \\
0 & 0 & 1 & T z_{i} \\
0 & 0 & 0 & 1
\end{array}\right]
\end{equation}
$$

这里只考虑刚性变换是因为和图像不同，3D目标的大小不会因为距离的变化而改变。还和STN不同的是，这里的训练时有监督的，使用预测目标的位置进行监督。

## 3D Transformer and Resampler

为了使得训练可以端到端，提出了该网络。点的刚性转换如下

$$
\begin{equation}
\left[\begin{array}{c}
x_{i}^{t} \\
y_{i}^{t} \\
z_{i}^{t} \\
1
\end{array}\right]=T\left(\Theta_{i}\right)\left[\begin{array}{c}
x_{i}^{s} \\
y_{i}^{s} \\
z_{i}^{s} \\
1
\end{array}\right]
\end{equation}
$$

转换的目标是为了将点云的中心转移到以目标为中心，直观的如下图所示。

## Localization and recognition

经过T-NET再预测刚性变换矩阵，以及BBox尺寸，最终在转换后的中心按预测出的尺寸得到目标的检测。

![](/assets/img/20210529/APF2.png)

## 具体细节

对于每个Cropped region，序列的长度$n=3$，也就是对一个序列做三次预测，如果这个里面的目标少于3个，再训练的时候就强制其回归到cropped region外面，来视为负样本，同时保持序列的固定长度。