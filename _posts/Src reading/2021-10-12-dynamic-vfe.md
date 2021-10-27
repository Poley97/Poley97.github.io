---
layout: post
title: 'MMDET3D DynamicVFE源码原理阅读'
date: 2021-10-12
author: Poley
cover: '../assets/img/mycat3.jpg'
tags: 源码阅读
---

本来想阅读dynamic VFE的论文，其用于网络MVX-Net中。但是其论文中并没有关于这部分的内容，因此只好直接阅读源码来理解这一机制。

mmdet3d中的voxel encoder一共有4种（截至目前），分别是
+ **HardVFE**
+ **HardSimpleVFE**
+ **DynamicVFE**
+ **DynamicSimpleVFE**

在命名上，Hard和Dynamic是相对的。Hard指使用固定的体素内点数，Dynamic是使用动态的体素内点数。

这些模块中，也具有和图像特征融合的功能，非常强大。

# HardVFE

使用固定体素网格大小的VFE。

其体素三维坐标的确定（这里指的是连续值，将一个体素视为一个点，而不是体素的Int坐标）。分为两种形式：

+ with_cluster_center : 使用体素内点的均值作为体素的坐标
+ with_voxel_center : 使用体素的中心点作为体素的坐标

注：程序中体素似乎是分两步执行的：
1. 给所有点分配一个体素坐标，并group
2. 将点按所述的体素的上述两种形式归一化
3. vfe

# HardSimpleVFE

不对体素网格内的点做归一化，直接平均作为体素的特征。也不行使用VFE(PointNet)，非常简单。

# DynamicVFE

整体流程与HardVFE一致，只是在点的分配上使用了 DynamicScatter() 函数。DynamicScatter() 函数用C++实现，暂时没有太看懂。



