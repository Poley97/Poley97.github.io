---
layout: post
title: 'BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird’s-Eye View Representation'
date: 2022-10-25
author: Poley
cover: '/assets/img/20221025/BEVFusionMIT.jpg'
tags: 论文阅读
---

本文是现在很火的关于BEV做多传感器感知融合的论文。目前有两篇均称为BEVFusion的工作，这篇是MIT产出的，还有一篇的北大产出的。两者差不多但略有区别。

本文的目的是，解决目标融合中的特征表达问题。使用BEV作为camera和lidar的统一特征表达形式。
传统的融合将两者统一到其中一个模态中，均有缺点。Lidar投影到camera处理，会损失准确的距离聚类信息（geometirc distortion）。反之，则相当于对camera中的密集信息做了下采样。同样造成了信息损失。如下图所示

![](/assets/img/20221025/BEVFusionMITF1.jpg)

即，传统方法中，问题在于点云的几何结构信息和图像密集的语义信息无法在融合特征的过程中得到同时保留。因此，本文创新，提出使用共享的BEV表征，同时保留几何结构信息和密集的语义信息（分别来自Lidar和Camera）。其网络结构如下所示

![](/assets/img/20221025/BEVFusionMITF2.jpg)

总的来说，本方法是一个多模态、多任务方法。融合多模态信息产生主干特征，根据不同的head实现不同的任务。目前来说，基于BEV的特征融合也逐渐成为了主流和热门领域（包括multi-camera和multi-sensor）

本方法的流程主要分为一下5步：
+ 单模态encoder；
+ 将各模态特征转换到BEV（图像基于LSS方法，Lidar则直接压缩到BEV）；
+ 解决view transformation和bev pooling带来的性能瓶颈（改进LSS的scatter方法）；
+ 使用基于卷积的bev encoder缓解局部特征的不对齐（由于图像深度估计的不准确或者内外参数的偏差）；
+ 使用不同的head完成特定的task（分割or检测）。

融合感知中面临的主要问题：
1. 相机和lidar的特征空间是不同的
2. 相机之间也存在显著的视角差异，导致相同的物体在不同的feature中可能对应完全不同的空间位置。这种情况下，elementwise feature fusion也不管用。

解决方法：使用统一的特征表达，应该具有：
1. 所有传感器特征可以轻松转换过去
2. 适用于不同的task。

即BEV特征表达，因此问题就变成如何高效的将camera和lidar的特征转换到BEV空间中
## Efficient Camera-to-BEV Transformation
这里主要使用的是LSS方法中提出的转换方法。根据深度估计的概率分布。将每一个feature map上的像素（而不是原始像素）都映射为其空间射线中的D个点（具体权重由估计的深度概率分布决定）。这样的操作形成一个非常密集的点云，之后将其全部投影到BEV上则实现了特征的转换。但是这其中有一个scatter的过程，将 HWD个点云映射到BEV空间中。由于图像的分辨率（像素数）本身相对点云数量就很多，HWD是一个很大的数字，这样的简单操作scatter就会付出了极大的计算代价。简单来说，是因为像素太多，即使feature map已经经过了降采样。（产生的点云太多）。

这里作者提出两个加速方法：
1. Precomputation：提前计算camera feature 和 bev grid的对应关系（由内外参决定，一般不会变化）。这可以将grid association的速度从17ms降低为4ms。
2. Interval Reduction： 做redution的时候需要一些对称函数，比如sum。传统做法（LSS）是通过prefix sum(前缀和)实现，导致了树形结构，多次内存读写和没有意义的中间值。

这里将每个grid分配到单独的cuda kernel上，单独计算，实现并行并减少内存读写。极大的提升了速度。如下图所示

![](/assets/img/20221025/BEVFusionMITF3.jpg)

## Fully-Convolutional Fusion

LiDAR2BEV的转换是非常自然的，就不再赘述了。两者都转换为BEV之后，可以实现elementwise level的操作以实现融合，比如concatnate。但是，由于深度估计上的不准确性，这种特征的concat一定是存在Misalign的，因此，作者这里还需要对融合后的特征进行进一步处理（卷积），增大感受野，缓解Misalignment。

## Multi-Task Head

上述产生的BEV feature map可以提供多重后续处理，比如centerhead或者detr head或者普通的det head来实现检测。or seg head以实现道路语义的分割。

## Experiments

本工作但是处于nuScenes的第一名。

![](/assets/img/20221025/BEVFusionMITT1.jpg)