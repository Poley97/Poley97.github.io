---
layout: post
title: '激光雷达数据集差异'
date: 2021-06-16
author: Poley
cover: ''
tags: 论文阅读
---

>这部分内容来自于ST3D论文的附录。

  # Overview

  不同数据及的差异包括：
  + LiDAR type
  + beam angles
  + point cloud density 
  + size
  + locations for data collection
![](/assets/img/20210617/ST3DTS7.png)
# Domain Difference and Systematic bias

## Lyft Annotation Discrepancies

Lyft数据集中，道路中间的目标更倾向于被标注，而道路两侧的反之。而Waymo数据集的标注最为丰富，使用Waymo预训练模型在Lyft上检测就会产生如下效果。

![](/assets/img/20210617/ST3DFS8.png)

检测到了大量Lyft中没标注的目标。导致很难评估模型在Lyft上的性能。

## Analysis of Domain Discrepancy

这些数据集之间的domain gap主要包括
+ Content gap：比如目标大小，由于不同的目标采集地点
+ Point distribution gap: 由于不同的雷达线数导致

## Systematic Bias on Pseudo Labels

这主要是由于 **Annotation style bias**，有些标注的框比较宽松，有些比较紧。

