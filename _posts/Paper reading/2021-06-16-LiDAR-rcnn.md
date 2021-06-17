---
layout: post
title: 'LiDAR R-CNN: An Efficient and Universal 3D Object Detector'
date: 2021-06-16
author: Poley
cover: '/assets/img/20210616/LRCNN.png'
tags: 论文阅读
---

>论文链接 ： https://arxiv.org/abs/2103.15297

本文使用了点处理方法来进行检测。但是发现了一个未被注意到的问题，**（在proposal中）单纯的提取点特征，比如PointNet，会使得学习到的点特征忽略size of proposal**。本文主要解决这个问题，并且得到了显著的提升。

# Introduction

作者发现一个很有趣的**size ambiguity problem**： 对于体素方法，一般是将proposal等分成固定的尺寸，而**PointNet仅仅将点特征做了aggregation而忽略了3D空间的尺寸**。

## Contuibution

+ 提出一个R-CNN style的两阶段检测器，基于PointNet。对于现有的3D检测器是**即插即用的（plug-and-play）**，并不需要重新训练。
+ 揭示了PointNet based检测器的**size anbiguity problem**。
+ 提升性能，并且可以实时。

# Methods

## Point-based R-CNN

假设现在已经有一种目标检测算法，可以提供perposal，我们的目标就是**refine the 7DOF bounding box parameters and scores of the proposals simultaneously**。
为了避免量化损失和插值，我们直接使用原始点云作为输入，而不是high-level DNN features.

### Input Features
为了得到更多的contextual points，扩大了proposal的范围，如下所示。同时对于proposal里的点做了归一化。以及坐标的旋转，使得x轴和目标方向对其。
![](/assets/img/20210616/LRCNNF1.png)

### Backbone

使用PointNet，如下所示。
![](/assets/img/20210616/LRCNNF2.png)

### Loss Function
和一般检测loss一样，只在方向上做了改动，防止方向损失明显大于其他损失。
$$
\begin{equation}
\begin{gathered}
\Delta \theta_{i}=\left(\theta_{i}^{g t}-\theta_{i}\right) \bmod \pi \\
\mathbf{t}_{i}^{o}=\left\{\begin{array}{ll}
\Delta \theta_{i}, & \Delta \theta_{i} \leq \frac{\pi}{2} \\
\Delta \theta_{i}-\pi, & \Delta \theta_{i}>\frac{\pi}{2}
\end{array}\right.
\end{gathered}
\end{equation}
$$

## Size Ambiguity Problem
![](/assets/img/20210616/LRCNNF3.png)

如上图所示，ambiguity主要是不同的proposal都可能包含同样一组点，这对于框的回归是不利的（这个问题之前的论文也提出过，具体是哪个忘了）。**上图的若干种方法都有自己的问题，比如（b）改变了形状，（c）不能包含位置信息，（d）外扩的体素可能是空的而不被利用，导致还是得不到边界信息，（e）边界信息直接和点特征结合，可以解决ambiguity问题（f）本文提出的方法**。

### Virtual Points
感觉和pvrcnn的grid是一样的，在空间中建立均匀分布的点来包含proposal的大小信息。

# Experiments

![](/assets/img/20210616/LRCNNF4.png)