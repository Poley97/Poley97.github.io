---
layout: post
title: 'Objects as Points'
date: 2021-12-22
author: Poley
cover: '/assets/img/20211222/ObjectasPoints.png'
tags: 论文阅读
---

CenterNet也是一篇很经典的Anchorless物体检测方法的论文。其优点与CornerNet类似，但是不同的是，由于这里检测的是物体的中心而不是物体的corner，因此可以进一步省去角点的匹配过程，提高了模型的速度。

![](/assets/img/20211222/ObjectasPointsF2.png)
本文的很多设置基本与CornerNet相同。
+ Heatmap设置：将目标中心点投影到heatmap上并给予一个合适的高斯分布的真值。对于量化损失，同样预测一个量化残差来解决。在Heatmap的损失上，使用和CornerNet一样的Focal loss的变种；
+ Size回归：这里直接回归目标的尺寸，而没有使用对数，而是对Size loss整体加上权重来平衡Loss的值；
+ 这套框架作者同样将其用于3D **检测和姿态检测中**，如下图所示
![](/assets/img/20211222/ObjectasPointsF4.png)