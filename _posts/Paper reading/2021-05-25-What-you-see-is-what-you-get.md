---
layout: post
title: 'What You See is What You Get: Exploiting Visibility for 3D Object Detection'
date: 2021-05-25
author: Poley
cover: '/assets/img/20210525/wyswyg.png'
tags: 论文阅读
---

> 论文链接 : https://openaccess.thecvf.com/content_CVPR_2020/html/Hu_What_You_See_is_What_You_Get_Exploiting_Visibility_for_CVPR_2020_paper.html

> 参考博客 :

# Introduction
目前的这些3D传感器，所采集到的数据，作者称其为2.5D数据，因为他们都不可避免的受到障碍物的影响。一个特定的物体在一个特定的位置，那么这个物体之后的沿着视线的区域就都被遮挡了。这也是为什么3D数据经常可以用2D表示的原因，所以称为2.5D。而PointNet等工作关注于真正的3D数据，比如mesh models上的采样点，这与现实的3D传感器产生了一定区别。

这种visibility信息也是传感器中包含的信息的一部分，将点云单纯的表示为$(x,y,z)$再做归一化就会损失这种visibility information。

本文贡献
1. 提出一种raycasting algorithms，有效计算voxel grid的 visibility；
2. 提出一种augmenting voxel-based networks with visibility 的方法；
3. 展示可以和SOAT结合。

![raycasting](/assets/img/20210525/WYSWYGF1.png)

![3D检测方法总体框架](/assets/img/20210525/WYSWYGF2.png)

# Related Work

略

# A General Framework for 3D Detection

3D Detecttion的主要方法（这里关注的）是BEV，BEV方法将3D点降为2D的就可以用2D卷积结构了。数据增强方面， copy-pastes是一个非常有效的方法，可以在vanilla PointPillars 上对增强的类别平均提升 9.1% 。另一种增强是时域增强，将多个雷达扫面的数据通过一些简单的方法融合到一张sence中（比如运动补偿），在vanilla PointPillars 提升可达 8.6% 。

## Compute Visbility through Raycasting

激光雷达的物理原理就是通过一个raycasting process，发射极光，接受回波，计算时间和距离，得到空间坐标$(x,y,z)$，但是这也表明坐标绝不仅仅包含上述主动感应得到的信息，同样包含了freespace along the ray of the pulse。

为了模拟LiDAR raycasting，我们计算点到原点的直线，然后计算其穿过的体素。这样就可以得到如上图所示的raycasting示意图。

但是计算体素和射线的相交（也就是体素的遍历）需要高效，因此参考一些方法中对空体素的忽略，本方法中只计算一组非空体素和ray的交集。具体方法是计算体素的六个axis aligned的表面哪个与射线相交，这个比较高效。当一个射线和一个非空体素相交，此射线就停止。和grid dim是线性的，非常高效。

有了这个raycasting 就可以进一步指导数据增强中的copy-pastes,使得粘贴过去的物体不会出现在视野盲区中。为了解决这个问题可以有两种方法

1. 把出现在视野盲区中的增强样本删了，称为culling
2. 把挡住增强样本的背景删了，称为drilling

上文的计算voxel 的Visibility仅仅计算了单帧的Visibility，对于连续的sweep，一个简单的想法就是，既然我们知道初始传感器的位置，我们就可以对所用帧都当做单帧处理，但是这类方法会造成比较大的time cost，作者采用了贝叶斯滤波对连续帧的visibility map就行预测。将visibility map转换为一个概率值，这样就把4D的数据转换为了3D。如下图所示，左图分别表示了单帧sweep的俯视图和对应的visiblity map，其中红色表示标记为occupy，蓝色为free space，灰色为unknown；图（b）则表示了采用贝叶斯滤波预测的多帧的点云俯视图和对应的visiblity，这里对于每一个vxole，越红则表示被占据的可能性就越大。


![3D增强结果对比](/assets/img/20210525/WYSWYGF3.png)

![visibility map](/assets/img/20210525/WYSWYGF4.png)

之后简单的visibility 和 Pointpillar的特征cat起来，得到了增强的特征，再进入下一步的处理，即可。性能有显著的提升。

![ARCH](/assets/img/20210525/WYSWYGF5.png)

# Experiments

![ARCH](/assets/img/20210525/WYSWYGEXP.png)