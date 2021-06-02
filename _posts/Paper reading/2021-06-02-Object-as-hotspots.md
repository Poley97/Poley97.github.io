---
layout: post
title: 'Object as Hotspots: An Anchor-Free 3D Object Detection Approach
via Firing of Hotspots'
date: 2021-06-02
author: Poley
cover: '/assets/img/20210602/OAH.png'
tags: 论文阅读
---

>论文链接 ：https://arxiv.org/abs/1912.12791

坐着发现除了将点云规则化，直接利用一个独立目标部分的点依旧对可以提供很多的关于目标语义的信息。因此提出了**Object as Hotspots(OHS)**，并且提出了一种新的anchor-free的方法以及一种新的assignment方法，来解决物体间点稀疏度不平衡的问题，防止网络向拥有更多点的object偏移。

本文利用compositional part-based models的思想，将点云的目标表示为**composition of their interior non-empty voxels**，将目标的非空体素视为一个**spots**,选择一个小子集作为**hotspots**。

Compostional models中，空间关系是很重要的，本文提出一个**spatial relation encoding**来强化hotspots之间内在的空间关系。

同时**hotspot策略也可以解决目标间点稀疏度不平衡的问题**。在KITTI中，目标中的的点数分布从**4874 to 1**，也就是特征不平衡问题。

OHS方法通过在每个object中选择有限数量的hotspot来balancing不同目标之间的正样本数量。同时，anchor-free没有认为定义的anchor-size，回归会更加困难，将其视为一个**regression target imbalance**问题，并通过一个**soft argmin**来解决。

## Object as Hotspots

### Hotspot Definition
Spots定义为非空体素。Hotspots是从中提取的一个子集，应该满足三个条件
1. 应该组成可分辨的parts来捕捉discriminatve features；
2. 应该在同一类别的物体中共享；
3. 应该最小化，使得物体中只有很少的LiDAR points被扫描，也就是说应该robust to objects with a small number of points。

最终只取M个hotspot，M由一个超参数和object中的voxel数量决定。
## HotSpot Network

![](/assets/img/20210602/OHSF1.png)

### Hotspot Classification

使用Focal loss作为损失。将Object内没有被选择为Hotspot的spot的梯度置零，使得他们不参与BP。
同样，loss的平均也使用hotspots和Non-hotspots的总数，除了gt BBox中的spot。

### Box Regression

回归hotspot到目标中心的xy偏移、z高度，以及lwh cos(r)和sin(r)。但是如之前所说，**Anchor-free的方法受到了regression target imbalance的影响**。因此使用了三个方法来削弱这个问题
1. 采用Log回归lwh，缩小了回归的绝对值
2. 采用cos(r),sin(r)回归，同样减小了回归的绝对值
3. 使用soft argmin来帮助回归 $d_x,d_y和z$，即将一个范围划分为n个bin，每个Bin有一个中心，则这个范围内任意的数可以表示为若干个中心的加权（类似于插值）。通过soft argmin预测这个权重，然后通过加权得到最后的预测值$t=\Sigma_{i}^{N}\left(S_{i} C_{i}\right)$

![](/assets/img/20210602/OHSF2.png)

### Spatial Relation Encoder

为了记录hotspot的空间关系，本文记录了在BEV中hotspot和目标中心点的相对位置关系，如上图中间所示，划分为四个区域，转换为一个分类问题。这是一个单独的Branch，这部分在推理过程中并不参与。
