---
layout: post
title: 'nuScenes: A multimodal dataset for autonomous driving'
date: 2021-06-21
author: Poley
cover: '/assets/img/20210621/nuScenes.png'
tags: 论文阅读
---


> 论文链接 ： http://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html

# Abstract

本文推出了nuScenes数据集，是第一个**搭在完全自动驾驶传感器的数据集**，包括：**6个谁昂头，5个雷达以及1个激光雷达，视野均为360度**。包含**1000个 scenes**，每个包含**20s**的数据，一起完整的3D BBOX标注，**包括 23个类别和 8 个属性**。标注量**7倍**与KITTI，图片数**100倍**与KITTI。

# Introduction
![](/assets/img/20210621/nuScenesF1.png)

在自动驾驶快速发展的今天，还缺少一种多模态的数据集来展示自动驾驶系统建立中面临的全面挑战。本数据集的提出就是为了解决这一点。

多模态传感器各自具有各自的优势和缺点。

+ **Cameras** : 提供准确的边缘，颜色，光照信息，但是缺少3D位置信息；
+ **Lidar** : 提供高度精确的3D位置，但是缺少语义（纹理）信息，范围一般50-150m；
+ **Radar** ：提供3D位置以及通过多普勒效应测量速度，但是会得到比激光雷达更稀疏、并且不如激光雷达准确的信息。尽管雷达已经被使用了十几年，但是还没有别的自动驾驶数据及提供radar数据。

![](/assets/img/20210621/nuScenesF2.png)

同时，目前还没有其他3D数据集提供了**属性标注**，比如行人的pose和车辆的状态。语义分割信息对于场景理解来说也是很重要的先验信息，比如一辆车一般出现在road上而不是sidewalk上。因此本数据也提供了**sematic maps**的标注。nuScense还提供所有传感器，所有场景下的360度环绕数据。

在环境上，包括了**夜间**和**雨天**，并将其加入了目标的属性当中。

评价标准上，提供了一套新的对于detection 和 tracking的评价标准，具体在公布的devkit中。

![](/assets/img/20210621/nuScenesT1.png)

# Related datasets

早期的自动驾驶数据都是2D标注，而多模态数据集的建立难度更大，主要包括多传感器的融合、同步以及校准。KITTI是这方面的先行者。

在初版nuScenes发布后，又有很多大规模自动驾驶数据集发布。其中，**Waymo**具有非常多的标注。而**Lyft L5 dataset**是最接近nuScense的，并且使用相同的格式，因此可以直接使用nuScenes devkit读取。

# The nuScenes dataset

## Drive planing

+ 驾驶地点 ： 美国波士顿、新加坡
+ 多样性 ： 穿过多个场景 vegetation, buildings, vehicles, road markings and right versus left-hand traffic
+ 时长 ： 15h
+ 距离 ： 242km
+ 速度 ： 平均16km/h
+ 环境 ： urban, residential, nature and industrial
+ 时间 ： day and night
+ 天气 ： sun, rain and clouds
![](/assets/img/20210621/nuScenesT2.png)

## Car setup
![](/assets/img/20210621/nuScenesF4.png)

## Sensor synchronization

当雷达扫过相机FOV的中心时，触发相机快门，以达成同步。

## Localization

为了解决GPS终端导致位置信息失效的问题，只做了高精度点云离线地图，来实现准确的定位，误差小于10cm。

## Map

语义分割使用语义地图的向量化实现，比光栅化具有更好的精度。如下所示。

![](/assets/img/20210621/nuScenesF3.png)

## Scene selection

包括 Such scenes include high traffic density (e.g. intersections, construction sites), rare classes (e.g. ambulances, animals), potentially dangerous traffic situations (e.g. jaywalkers, incorrect behavior), maneuvers (e.g. lane change,
turning, stopping) and situations that may be difficult for an AV.

## Data annotation

数据关键帧采样为2Hz,23个目标类，3D BBox包括x, y, z, width, length, height and yaw angle。

![](/assets/img/20210621/nuScenesF5.png)

## Annotation statistics

每个关键帧**平均 7个行人，20辆车**。超过40k的关键帧来自四个不同的位置：**Boston: 55%, SG-OneNorth: 21.5%, SG-Queenstown: 13.5%, SG-HollandVillage: 10%) with various weather and lighting conditions (rain: 19.4%, night: 11.6%**。体现出严重的数据不平衡，标注最少和最多的目标数比例达到**1:10k**，这个数字在KITTI中是**1:36**。即**nuSence是一个长尾数据集**。标注数据在最远80m出可以保持100个激光雷达点云，在最近3m处可以达到12k个点云（激光雷达作用距离标称70m）。

# Tasks & Metrics
## Detection
和传统的IOU判断样本是否匹配不同，这里使用地平面上的2D中心距离阈值来判断（也可以用IOU），因为对于一些小目标比如行人来说，一点点的偏移错误就会让IOU变为0。 mAP的计算对所有类别，所有阈值下的AP做平均得到，如下所示。

![](/assets/img/20210621/nuScenesE1.png)

其余的度量如下：
Average Translation Error (ATE) is the Euclidean center distance in 2D (units in meters). Average Scale Error (ASE) is the 3D intersection over union (IOU) after aligning orientation and translation (1 − IOU). Average Orientation Error (AOE) is the smallest yaw angle difference between prediction and ground truth (radians).Average Velocity Error (AVE) is the absolute velocity error as the L2 norm of the velocity differences in 2D (m/s). Average Attribute Error (AAE) is defined as 1 minus attribute classification accuracy (1 − acc).

## Trackting 
略

# Experiments 
略