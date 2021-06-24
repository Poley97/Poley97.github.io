---
layout: post
title: 'Scalability in Perception for Autonomous Driving: Waymo Open Dataset'
date: 2021-06-21
author: Poley
cover: '/assets/img/20210621/Waymo.png'
tags: 论文阅读
---



> 论文链接 ： http://openaccess.thecvf.com/content_CVPR_2020/html/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.html

# 概述
针对现有自动驾驶数据集规模和环境变化不多的问题，提出了新的Waymo数据集。包含1150 scenes ，每个时间长度20s。

# Introduction

Waymo数据集是现在最大，最多样化的多模态自动驾驶数据集。由多个高分辨相机和多个高质量激光雷达得到数据。同时，数据集中地理环境的多样化也比其他任何自动驾驶数据集强。数据在多个城市中被记录，每个城市都有很大的地理信息的差异。**我们证明这些不同的地理信息导致了数据间的domain gap，这使得这个数据集可以有机会用来研究domain adaptaion领域。**

标注数量：
+ LiDAR box : 12 million
+ Camera box : 12 million
+ LiDAR object tracks : 113k
+ camera imgae tracks : 250k

# Related Work

此处指出 nuScense数据集的缺点在于其有限的激光雷达传感器质量（因为是32线），一帧只有32K个点。并且地理环境多样性也有限,由于其有效覆盖范围只有5km。（但是可能v1.0的 nuScense不是这样的，从它的论文来看）。

![](/assets/img/20210621/WaymoT1T2.png)

![](/assets/img/20210621/WaymoT3.png)

![](/assets/img/20210621/WaymoF1.png)

# Waymo Open Dataset
## Sensor Specifications

车上有5个激光雷达和5个高分辨率针孔相机。其具体角度信息如上表上图所示。

## Coordinate Systems

+ Global frame：东北天对应xyz
+ Vehicle frame： 前左上对应xyz
+ Sensor frame: 如上图所示
+ Image frame: x width y height
+ LiDAR Spherical coordinate system: 将雷达坐标转换为极坐标，略。

## Ground Truth Labels

标注使用 7自由度3D BBOX编码 cx, cy, cz, l, w, h, $\theta$，加tracking ID的形式。图像编码4 DOF cx, cy, l, w + tracking ID。和KITTI类似，采用了两个等级来表示difficulty ratings。其标准来自于手工标注者以及目标的统计信息。
![](/assets/img/20210621/WaymoF3.png)
![](/assets/img/20210621/WaymoF4.png)
![](/assets/img/20210621/WaymoT4.png)

## Datast Analysis
![](/assets/img/20210621/WaymoT5.png)

+ 数据覆盖面积 $40km^2$ in Phoenix, $36km^2$ combined in San Francisco and Mountain View。
+ 标注量 The dataset has around 12M labeled 3D LiDAR objects, around 113k unique LiDAR tracking IDs, around 12M labeled 2D image objects and around 254k unique image tracking IDs.


# EXPERIMENTS

略，下表是关于域迁移的结果。

![](/assets/img/20210621/WaymoT8T9.png)
