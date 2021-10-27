---
layout: post
title: 'New Frontiers in Perception for Scalable Autonomous Driving'
date: 2021-09-07
author: Poley
cover: '/assets/img/20210907/PerceptionAD1.png'
tags: Lectures
---

>视频链接： 来自深蓝学院 https://www.shenlanxueyuan.com/open/course/111/lesson/100/liveToVideoPreview

主题：**如何让自动驾驶系统中的感知系统更加的Scalable**

目前主流的无人车系统中，在工业界更多的还是采用模块化设计

![](/assets/img/20210907/PerceptionAD2.png)

其中BP是行为预测，Planner是规划。

MAP可以为Perception提供很多先验知识，因此地图信息对于感知来说也是很重要的。

Perception的输入应使用尽可能多的互补的输入，如下所示

![](/assets/img/20210907/PerceptionAD3.png)

输出 **Repersentation of Enviroments**。

常见的感知任务如下

![](/assets/img/20210907/PerceptionAD4.png)

**Scalability in Perception**

![](/assets/img/20210907/PerceptionAD5.png)

其中，三个方面是学术界主要关注的，而两个是工业界更关注的。
![](/assets/img/20210907/PerceptionAD6.png)

# 介绍工作

## SPG
![](/assets/img/20210907/PerceptionAD7.png)

核心思想：**通过在检测之前进行3D形状补全，来实现更可靠更准确的Detection**

![](/assets/img/20210907/PerceptionAD8.png)

## RSN

![](/assets/img/20210907/PerceptionAD9.png)

核心思想：**通过预先进行前景分割来去除大量背景点，做高效的BEV检测。**


# Offborad

![](/assets/img/20210907/PerceptionAD10.png)

自动数据标注方法

![](/assets/img/20210907/PerceptionAD11.png)

为了解决标注成本高的问题，提出使用3D AUTO Labeling+ Partial Manual Labeling

![](/assets/img/20210907/PerceptionAD13.png)

理由： offboard 可以一次得到更多的信息，并且没有计算能力的限制。如下所示

![](/assets/img/20210907/PerceptionAD14.png)

# Data Flexibility

目标：生成真实图像，提高Data Flexibility 来帮助算法进行学习。
![](/assets/img/20210907/PerceptionAD15.png)

![](/assets/img/20210907/PerceptionAD16.png)

利用3D数据中对物体的平移旋转比较简单的特点，可以进行数据增强。进而使用GAN网络来生成增强后的点云的对应的image。