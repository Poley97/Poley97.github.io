---
layout: post
title: 'Argoverse 3D Tracking and Forecasting with Rich Maps'
date: 2023-3-13
author: Poley
cover: '/assets/img/20230311/Argoverse.jpg'
tags: 论文阅读  
---

本文章主要是对Argoverse数据集的说明。其数据可视化如下图所示

![](/assets/img/20230311/ArgoverseF1.jpg)

![](/assets/img/20230311/ArgoverseF2.jpg)

主要配置，Lidar + 7cam 覆盖360度，期货中包括一个例题相机。提供6自由度姿态。和其他数据集最大的不同，即提供丰富的语义原数据，包括车道、可行驶区域，地面高度以及物体的属性等其他信息。baseline实验证明这些lane-level数据可以有效的减少追踪/预测的错误。本数据集最大的优势似乎在tracking上，有超过300k的vehicle轨迹。同时，提供了map数据，可以支持map-based perception算法的研究。同时也支持对map的benchmark。

在数据规模上，标注的面积和nuscenes近似，但是由于帧率更高，因此实际上数据集比nuscenes大很多，具体对比如下

![](/assets/img/20230311/ArgoverseT2.jpg)

笔者这里主要用做检测目的，因此对tracking等问题不是很关系，这里略过。数据集的传感器配置为：
+ Lidar: VLP32*2，叠着放，范围200m
+ Cam：7个，分辨率1920*1200，30hz。两个立体相机，2056*2464，5Hz。
+ 姿态：组合导航，6DOF。
+ 坐标系：车辆坐标系的中心在车辆后轴的中心。具体如下图所示

![](/assets/img/20230311/ArgoverseF3.jpg)

其论文中也没有给出检测任务的baseline，故在此略过。