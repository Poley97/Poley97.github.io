---
layout: post
title: 'PillarNet: Real-Time and High-Performance Pillar-based 3D Object Detection'
date: 2022-10-11
author: Poley
cover: '/assets/img/20221008/PA.jpg'
tags: 论文阅读
---

本文发表于ECCV2022，提出一种简单有效的3D检测网络。主要贡献：1、探明了pillar-based和voxel-based的性能差异；2、提出一种新的pillar-based结构，同时兼顾效率和性能。

作者对pillar-based和voxel-based网络的结构进行了分析：Voxel-based网络首先经过3D卷积，再变为BEV，使用neck 2D卷积处理。而pillar-based直接就进入neck 网络进行处理。同时由于编码器中的3D稀疏卷积，使得在BEV空间中聚合多分辨率特征变得很困难。但是不管是voxel还是pillar的方法，都使用了在BEV空间中聚合多尺度特征的操作。如下图所示
![](/assets/img/20221008/PillarnetF2.jpg)

作者提出，pillar-based的编码器太简单，导致性能不足。这里和传统2D处理一样，使用一个多stage的encoder比如VGG或者ResNet。

同时，多次下采样使得网络对piilarsize不再敏感。只需要根据pillarsize选择对应的stage(下采样率)即可，助输出特征图尺寸的初始化的psudo-image投影尺度的decouple。

作者也发现，现有的融合方法大多依赖于中间的BEV表征。因此本方法有望无缝衔接到多模态框架中以实现更高性能。

目前现有的方法在认为3D稀疏卷积有表达优势的情况下，基于pillar的方法通常关注于Pillar如何从raw points中提取特征或者设计复杂的多尺度策略。这带来额外的延迟并且仍然在性能上落后于3D voxel-based的方法。

本文的主要贡献在于。作者提出，pillar-based的主要性能瓶颈在于：
**稀疏encoder网络对于空间信息的学习，以及neck有效的空间 -语义特征融合的能力。**

对应的，作者提出两个设计：使用稀疏2D卷积取代3D卷积，作为pillar的encoder。neck则负责空间-语义信息融合。如下图所示

![](/assets/img/20221008/PillarnetF3.jpg)

在neck的选择上，作者提出三种选择，分别是照搬SECOND，以及两种从backbone提取部分特征的方式，如下图所示

![](/assets/img/20221008/PillarnetF4.jpg)

最后，作者提出了一种Orientation-Decoupled IoU Regression Loss，用于目标框的回归。首先，对分类得分，使用iou得分对分类得分进行指数加权，如下

$$
\begin{equation}
\hat{S}=S^{1-\beta} * W_{\mathrm{IoU}}^\beta
\end{equation}
$$

在回归损失上，IoU based的回归损失是一种常用的方法。但是由于3D IoU的复杂计算，2D中的一些基于IoU的回归损失会降低训练速度，同时，可能产生一些负面的影响（由于旋转的存在）。在不同误差的情况下，实际上旋转的局部最优并不在gt上（或者是有一段flatten）。如下图所示


![](/assets/img/20221008/PillarnetF5.jpg)

因此本文拓展了这些IoU损失（IoU/GIoU/DIoU），将其与旋转方向decouple实现更好的优化。但是具体的方法没在论文里说？可能需要看一下代码。

最后，其在nuScenes和Waymo上都取得了惊人的高性能。
![](/assets/img/20221008/PillarnetT1.jpg)
![](/assets/img/20221008/PillarnetT2.jpg)
