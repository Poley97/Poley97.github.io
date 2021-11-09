---
layout: post
title: 'PointPainting: Sequential Fusion for 3D Object Detection'
date: 2021-11-05
author: Poley
cover: '/assets/img/20211101/PointPainting.png'
tags: 论文阅读
---

相机和激光雷达是可以提供互补的信息，但是出乎意料的是，目前数据融合方法没有lidar only的方法表现好，这说明这里存在一个Gap。本文通过PointPainting(点云上色)来弥补这一点。

目前来说，BEV方法的表现显著优于局域range image的，这是因为BEV具有尺度不变形，并且阻挡小，没有深度模糊效应（在对range-image做2D卷积的时候产生）。

以往的fusion方法大题可以分为四类：**object-centric fusion, continuous feature fusion, explicit transform and detection seeding.**

+ Object-centric fusion ：当得到proposal之后，从多模态中进行roi-pooling;
+ Continuous feature fusion: 在backbone上就进行了两者的融合;
+ Explicit transform: 显式的将图像转换成bev表示，并在此进行融合。这需要建立伪点云能实现;
+ detection seeding： 将图像中的检测结果作为先验，之后再进一步在点云中检测。比如Frustrum PointNet。

本文题出的PointPainting是一种图像和点云的数据融合方法，可以移植到任何lidar-only的方法中。其流程如下所示：

![](/assets/img/20211101/PointPaintingF2.png)

### Image Based Semantics Network

直接使用DeepLabv3+得到。使用分割有两点好处：1、分割是一个相对简单的任务，因此在数据融合中不太影响推理速度；2、使用分割做数据融合可以使得PointPainting同时从分割和3D目标检测的发展中受益。

### PointPainting
流程如下所示
![](/assets/img/20211101/PointPaintingA1.png)

方法非常简单，先对图像做完分割，再将点云投影到图像像素上，并获取所在像素的语义分割结果，作为点云的额外特征，送入任意的liday only算法中

## Experiments

![](/assets/img/20211101/PointPaintingT2.png)
