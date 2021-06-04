---
layout: post
title: 'Voxel R-CNN: Towards High Performance Voxel-based 3D Object Detection'
date: 2021-06-02
author: Poley
cover: '/assets/img/20210602/VRCNN.png'
tags: 论文阅读
---

>论文链接 ： https://arxiv.org/abs/2012.15712v2

一般认为点方法具有跟高的定位精度，但是要付出更多的计算代价。而体素化由于将点云划分到网格里，造成了信息的损失，往往精度不如点方法。但是本文作者提出了一个不相同的观点，**其认为原始点的精确位置并不是高性能3D目标检测所必须的，而粗粒度的体素同样可以提供足够的检测精度**。

# Introduction
目前3D点云的处理方法大致可以分为两类，**voxel-based 和point-based**，一般来说，voxel-based具有更快的速度，而point-based具有更高的精度。**同时，point-based的方法占据着多个benchmarksd的top rank，这样形成了一种普遍的看法，就是原始点云的位置信息是进行准确目标定位的关键**。

作者发现，**voxel-based的方法一般都在bird-eye-view(BEV)上进行目标检测的任务，而point-based则相反，一般都依靠提取点的特征来保存3D结构上下文信息，并通过点特征进一步做refinement**。

作者认为，**阻碍voxel-based方法的主要问题是在将3D feature转换到BEV时没有保存3D structure context**。

根据上述问题，作者提出的Voxel R-CNN,主要由三部分组成
1. A 3D backbone network
2. A 2D backbone network followed by the Region Proposal Network(RPN)
3. A detect head with a new coxel RoI pooling operation

# Reflection on 3D Object Detection

## Revisiting and Analysis

首先复习了SECOND和PV-RCNN两种算法。这两者主要有两个区别：

1. SECOND是单阶段方法，PV-RCNN是两阶段算法；
2. PV-RCNN中的keypoints保留了3D的结构信息，而SECOND直接在BEV上进行预测。

为了验证作者的想法，作者给SECOND加了一个detect head，性能提升了0.6%，但是仍和PV-RCNN相差较大，如表1所示。这即说明了BBox refinement的作用，也验证了 BEV特征表示的容量（能力）是有限的。同时，如表2所示所示，VSA花费了大量的时间，所以导致PV-RCNN的速度慢于SECOND。

![](/assets/img/20210602/VRCNNT1T2.png)

综上，找到了PV-RCNN相比SECOND速度慢而性能好的原因。得到结论：

1. **3D strucutre is of significant importance for 3D object detectors**;
2. **The point-voxel feature interaction is time-consuming and affects the detector;s efficiency**.

这就很自然的引出了作者的设计，Voxel-RCNN。

# Voxel R-CNN Design
![](/assets/img/20210602/VRCNNF2.png)
## Voxel RoI pooling
为了直接对3D voxel feature volumes 的spatial context 做 aggregation，提出了Voxel RoI pooling。

### Voxel Query
提出了一种Voxel Query，利用Voxel 分布的规则特征，可以高效的group voxels,**相当于省去了离散点做近邻搜索的时间**,如下所示。
![](/assets/img/20210602/VRCNNF3.png)

利用体素的数据的规则性可以很方便的提取近邻，和一般的近邻搜索不同的是，这里使用的是**曼哈顿距离**，故可以得到上图中的近邻范围。这样对于N个非空体素，计算复杂度仅为$O(N)$，显著提高了效率。

### Voxel RoI Pooling Layer
首先将region proposal into $G \times G \times G$的网格。由于3D特征非常的系数，不能直接用maxpooling，在region proposal上。**所以使用融合邻域体素特征的方式来得到sub-voxel的特征（邻域体素特征经过PointNet Module）。**同时，在提取特征时，使用了backbone最后两个stages的特征，以及不同的邻域范围（曼哈顿距离阈值）。

### Accelerated Local Aggregation
![](/assets/img/20210602/VRCNNF4.png)
参考了一些其他文章的加速方法，大概意思就是将坐标和特征分开处理了？

## Backbone and Region Proposal

同样先用3D网络，再转换成2D BEV做RPN。

## Detect Head

使用MLP，并无特殊。

## Training Objectives

### RPN loss
继承了VoxelNET的Loss,如下

$$
\begin{equation}
\mathcal{L}_{\mathrm{RPN}}=\frac{1}{N_{\mathrm{fg}}}\left[\sum_{j} \mathcal{L}_{\mathrm{cls}}\left(p_{i}^{a}, c_{i}^{*}\right)+\mathbb{1}\left(c_{i}^{*} \geq 1\right) \sum_{i} \mathcal{L}_{\mathrm{reg}}\left(\delta_{i}^{a}, t_{i}^{*}\right)\right]
\end{equation}
$$

## Losses of detect head

分配标签的方法使用阈值+IOU的方式
$$
\begin{equation}
l_{i}^{*}\left(\operatorname{IoU}_{i}\right)=\left\{\begin{array}{ll}
0 & \mathrm{IoU}_{i}<\theta_{L}, \\
\frac{\mathrm{lo} U_{i}-\theta_{L}}{\theta_{H}-\theta_{L}} & \theta_{L} \leq \mathrm{IoU}_{i}<\theta_{H}, \\
1 & \mathrm{IoU}_{i}>\theta_{H},
\end{array}\right.
\end{equation}
$$

分类使用交叉熵
$$
\begin{equation}
\begin{aligned}
\mathcal{L}_{\text {head }}=& \frac{1}{N_{s}}\left[\sum_{i} \mathcal{L}_{\text {cls }}\left(p_{i}, l_{i}^{*}\left(\operatorname{IoU}_{i}\right)\right)\right.\\
&\left.+\mathbb{1}\left(\operatorname{IoU}_{i} \geq \theta_{r e g}\right) \sum_{i} \mathcal{L}_{\text {reg }}\left(\delta_{i}, t_{i}^{*}\right)\right]
\end{aligned}
\end{equation}
$$