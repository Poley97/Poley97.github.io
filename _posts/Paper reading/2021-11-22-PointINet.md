---
layout: post
title: 'PointINet: Point Cloud Frame Interpolation Network'
date: 2021-11-22
author: Poley
cover: '/assets/img/20211122/PointINet.png'
tags: 论文阅读
---

由于LiDAR的采集频率一般远低于其他传感器的帧率，比如相机（相机30-100Hz，激光雷达10-20Hz，IMU 可以到200Hz）。本文提出一种点云帧插值算法，PointINet，通过估计双向的3D scene flow来实现准确的插值。

在图像领域，从低帧率视频插值得到高帧率视频的技术已经很普遍（比如30Hz to 240Hz），但是点云中这种方法还没有怎么得到发展。

3D场景流(3D Scene flow estimation)可以视作2D光流在3D场景中的拓展，表示点的运动场。
## Point cloud frame interpolation

![](/assets/img/20211122/PointINetF2.png)

网络的整体结构如上所示。

### Point cloud warping
给定两个点云$P_0$和$P_1$，这个模块的任务是预测这两幅点云中的点在$t$时刻的位置$P_{0,t},P_{1,t}$。这里的关键是估计两帧之间的scene flow。这里使用的是FlowNet3D来估计两帧之间的双向3D scene flow，假设两个连续帧之间的运动是线性的，因此两者到t时刻的3D scene flow可以表示为
$$
\begin{equation}
\begin{aligned}
&F_{0 \rightarrow t}=t \times F_{0 \rightarrow 1} \\
&F_{1 \rightarrow t}=(1-t) \times F_{1 \rightarrow 0}
\end{aligned}
\end{equation}
$$

进而可以得到两帧点云分别在t时刻的位置为
$$
\begin{equation}
\begin{aligned}
&\hat{P}_{0, t}=P_{0}+F_{0 \rightarrow t} \\
&\hat{P}_{1, t}=P_{1}+F_{1 \rightarrow t}
\end{aligned}
\end{equation}
$$

### Points fusion
在2D视频插帧中，由于视频（图像）是网格化的数据，因此这一步主要用于处理由于像素运动带来的网格遮挡或者空白。但是由于点云是非结构化并且无序的，因此点云的fusion要复杂得多。

这里的方法是自适应的从两个t时刻的点云中采样点，并以采样点为中心做kNN，之后使用一个attentive points fusion module来得到最终的点云。

#### Adaptive sampling

这里指根据时间来动态的调整从两幅点云中采样的比例。即$N_1=(1-t)\times N,N_2 = t \times N$,最后组成有$N$个点的点云。

#### Adaptive kNN cluster
对于上述每个采样点，我们在两幅点云中搜索以采样点为中心的kNN。和上述采样一样，这里使用同样的方法将K个采样点分为两份，在两幅点云中采集。邻域中的每个点云使用相对位置编码。

#### Attentive points fusion
![](/assets/img/20211122/PointINetF3.png)

非常简单，不再赘述。

## Experiment
![](/assets/img/20211122/PointINetF4.png)

个人认为这个论文中最难的部分在于场景流估计，然而在这篇文章中，这部分中并没有任何创新点。故本人认为这篇文章价值有限。
