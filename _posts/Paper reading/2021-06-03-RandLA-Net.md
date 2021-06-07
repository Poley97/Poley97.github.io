---
layout: post
title: 'RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds'
date: 2021-06-03
author: Poley
cover: '/assets/img/20210603/RandLANet.png'
tags: 论文阅读
---

>论文链接：http://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_RandLA-Net_Efficient_Semantic_Segmentation_of_Large-Scale_Point_Clouds_CVPR_2020_paper.pdf

# Introduction

![](/assets/img/20210603/RandLANetF1.png)
PointNet提出了直接处理3D点云的方法，直接通过MLPs。这样的计算效率很高，但是缺乏捕捉更宽泛context information的能力，为了学习到更充分的局部结构，主要有以下几种结构：

1. **Neighbouring feature pooling**
2. **Graph message passing**
3. **Kernel-based convolution**
4. **Attention-based aggregation**

但是上述这些方法都都被限制为处理很小的3D点云，比如4000个点或者1*1米的blocks，而不能直接处理大规模的点云，比如上百万个点以及200*200m的范围。这主要由以下三个原因造成

1. **这些方法使用的point-sampling方法computationally expensive或者 memory inefficient**,比如对1M个点进行FPS下采样为原来的10%，需要200秒的时间；
2. **目前的local feature learners通常依赖于computationally expensive kernelisation or graph construction**，因此很难处理大规模点；
3. **大尺度点云通常包含上百个点云，现有的local feature learners要不没有能力捕捉complex structure，要不不够高效，由于他们有限的感受野**。

本文着重于设计一种内存和计算都高效的网络结构来直接处理大规模3D点云。综上，主要要解决两个问题：

1. 采样的高效；
2. 有效的local feature learner。

作者发现，**random sampling**是高效采样的关键，但是**这样会随机的丢失关键信息，尤其是对于只有稀疏点的目标**，所以有提出了一种**local feature aggreagation module**来解决这个问题，其用于有效的学习complex local structures by progressively increasing the receptive field size in each neural layer。首先使用了一个**local spatial encoding(LocSE)**来保留局部几何结构，再使用**attention pooling**来自动保留有用的局部特征。最后将两者结合为一个**dilated residual block**。

# Related Work

略

# RandLA-Net
## Overview
如下图所示，输入点在网络的前进过程中逐步并且高效的下采样，而不是丢失有用的信息（点特征）。
![](/assets/img/20210603/RandLANetF2.png)

## The quest for efficient sampling

### Heuristic Sampling

1. **Farthest Point Sampling (FPS):** 计算复杂度$O(n^2)$，不适合大场景。处理1M点需要约200s，在单Gpu上。收敛性好。

2. **Inverse Density Importance Sampling (IDIS):** 计算复杂度$O(N)$,处理1M点需要约10s，在单Gpu上。对噪声敏感。
3. **Random Sampling (RS):** 计算复杂度$O(1)$，处理1M点需要约0.004s，在单Gpu上。

### Learning-based Sampling

1. **Generator-based Sampling (GS):** 其学习产生一小组点云来近似表示原始点云集合。但是在inference时一般使用FPS来将生成子集和原始集合匹配。实验中，处理1M点采样10%需要约1200s。
2. **Continuous Relaxation based Sampling (CRS):** 使用重参数化的trick来将采样操作松弛到一个连续域中for end-to-end training。每个采样点通过素有点云的一个加权求和来学习。这导致一个超大的矩阵，造成不可承受的内存负担。实验中，处理1M点采样10%需要约300GB内存。
![](/assets/img/20210603/RandLANetF3.png)

3. **Policy Gradient based Sampling (PGS):** 以马尔科夫决策概率的方式来采样，但是面对大量点云的时候由于，由于搜索空间太大，学习到的概率具有很大的方差，并不高效，而且也不好收敛。

## Local Feature Aggregation

### Local Spatial Encoding

首先将点特征和点坐标cat起来，然后找到一个点的K个邻居，将他们一起处理。对应点的特征就会关注到他们的相对位置关系。如上图所示，做完近邻搜索后分为两个分支

1. **Relative Point Position Encoding**:其特征编码方法如下
$$
\begin{equation}
\mathbf{r}_{i}^{k}=M L P\left(p_{i} \oplus p_{i}^{k} \oplus\left(p_{i}-p_{i}^{k}\right) \oplus\left\|p_{i}-p_{i}^{k}\right\|\right)
\end{equation}
$$

2. **Point Feature Augmentation**
   邻域点的特征就普通的经过一个MLP得到，然后和上述特征相加，得到**增强的点特征**。


### Attentive Pooling

现有的方法主要通过max/mean pooling来hard integrate（硬集成），这样造成了信息损失。取而代之的本文采用attention来自动的学习重要的局部特征。

Attention模块简单的通过一个MLP和softmax对原点特征进行加权，然后以求和的方式进行融合，如下。
$$
\begin{equation}
\mathbf{s}_{i}^{k}=g\left(\hat{\mathbf{f}}_{i}^{k}, \boldsymbol{W}\right)
\end{equation}
$$

$$
\begin{equation}
\tilde{\mathbf{f}}_{i}=\sum_{k=1}^{K}\left(\hat{\mathbf{f}}_{i}^{k} \cdot \mathbf{s}_{i}^{k}\right)
\end{equation}
$$

### Dilated Residula Block

通过连续的Local Spateial Encoding 和 Attentive Pooling，以及skip connection组成一个block。数据随着不断的降采样和融合特征，使得剩余的点感受野越来越大，如下所示。

![](/assets/img/20210603/RandLANetF4.png)

# Experiments
首先，在速度和处理大规模点云的能力上，远超现有方法，如下所示。
![](/assets/img/20210603/RandLANetT1.png)

在性能指标上，同样超过现有方法，效果很好。

![](/assets/img/20210603/RandLANetT3.png)

![](/assets/img/20210603/RandLANetF6.png)