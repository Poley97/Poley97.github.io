---
layout: post
title: 'A Closer Look at Local Aggregation Operators in Point Cloud Analysis'
date: 2021-06-04
author: Poley
cover: '/assets/img/20210604/LAO.png'
tags: 论文阅读
---

论文链接： https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680324.pdf

本文首先回顾了多重local aggregation operators，并且使用相同的深度残差结构来研究他们的性能。并提出了**Position Pooling(PosPool)**,相比现有的复杂方法具有类似甚至更好的性能。

# Introduction

现有的local aggregation operators主要可以粗略的分为3类：

1. **Point-wise MLP based**: 主要使用一个小的PointNet来实现。
2. **Pseudo grid feature based**：使用一个在预定义网格位置上的pseudo features，然后学习这些位置上的权重，就像普通的卷积。
3. **Adaptive weight based**： 通过加权求和来aggregate 邻域特征。

由于目前的方法都不是单独提出的，所以他们都具有不同的网络结构，不好公平的比较。并且大多数现有的aggregation layers都是基于浅层网络的，并不清楚他们是否可以在深层网络上见效。

本文选用了三个数据集，他们具有不同的tasks，场景以及数据规模。分别是：**ModelNet40,S3DIS and PartNet**。作者将多重aggregation在这些数据上以相同的网络结构做公平的比较，发现他们的性能都差不多，引出一个问题。**是否需要这么复杂的local aggregation computation？**

于是，作者提出了**PosPool**，直接使用邻域点的特征和相对坐标做**element-wise multiplication**，展现了不错的性能，证明不需要那么复杂的操作来做aggregation。

# Overview of Local Aggregation Operators

## General Formulation
$$
\begin{equation}
\mathbf{g}_{i}=R\left(\left\{G\left(\Delta \mathbf{p}_{i j}, \mathbf{f}_{j}\right) \mid j \in \mathcal{N}(i)\right\}\right)
\end{equation}
$$

其中，$G$是一个变换函数，然后通过一个reduction function $R$来aggregation所有的数据。使用这个通式，可以几种local aggregation operators的表达式：

1. **Point-wise MLP based**
$$
\begin{equation}
G\left(\Delta \mathbf{p}_{i j}, \mathbf{f}_{j}\right)=\operatorname{MLP}\left(\operatorname{concat}\left(\Delta \mathbf{p}_{i j}, \mathbf{f}_{j}\right)\right)
\end{equation}
$$

2. **Pseudo grid feature based**
$$
\begin{equation}
\mathbf{f}_{i, k}=\sum_{j \in \mathcal{N}(i)} \max \left(0,1-\frac{\left\|\Delta \mathbf{p}_{j k}\right\|_{2}}{\sigma}\right) \cdot \mathbf{f}_{j}
\end{equation}
$$
$$
\begin{equation}
G\left(\Delta \mathbf{p}_{i k}, \mathbf{f}_{i, k}\right)=\mathbf{w}_{k} \odot \mathbf{f}_{i, k}
\end{equation}
$$
3. **Adaptive weight based**

$$
\begin{equation}
G\left(\Delta \mathbf{p}_{i j}, \mathbf{f}_{j}\right)=H\left(\Delta \mathbf{p}_{i j}\right) \odot \mathbf{f}_{j}
\end{equation}
$$


# Benchmarking Local Aggregation Operators in Common Deep Architecture
![](/assets/img/20210604/LAOF1.png)

首先，提出一种深度残差架构，如上图所示。

提出了一种5-satge的深度残差架构。这种深度网络在小数据上可能会造成性能的下降，比如ModelNet40，但是在大数据集上一般会有明显的心梗提成，比如PartNet。

降采样的方式使用体素降采样，每个satge使用越来越大的体素网格（两倍大），这样点云就会不断的稀疏下去。

同时考虑不同的模型容量，通过改变网络深度（Block repeating factoe $N_r$），width($C$)，以及bottleneck ratio($\gamma$)。

在邻域的检索上，使用的是**ball radius method**，其产生的结果具有相对**KNN**更加均匀的密度。

![](/assets/img/20210604/LAOT1.png)

两种baseline所代表的aggregation方式如下

$$
\begin{equation}
\mathbf{g}_{i}=\mathbf{f}_{i}
\end{equation}
$$

$$
\begin{equation}
\mathbf{g}_{i}=R\left(\left\{\mathbf{f}_{j} \mid j \in \mathcal{N}(i)\right\}\right)
\end{equation}
$$

## Performance Study on Point-wise MLP based Method


![](/assets/img/20210604/LAOT2.png)

结果显示，**FC层使用一侧效果最好，猜测可能是多层FC相当于对单点做一个point-wise transformation，可能不利于优化**。输入上，**各种对结果的贡献都差不多**。Reduction function上，**MAX Pooling最好**。

## Performance Study on Adaptive Weight based Method

![](/assets/img/20210604/LAOT3.png)

结果显示，**FC层使用一侧效果最好，猜测可能是多层FC相当于对单点做一个point-wise transformation，可能不利于优化**。输入上，**只使用相对位移效果最好，其他的都不好，可能是hi由于阻碍了自适应学习从相对位移中的学习**。Reduction function上，**MAX AVG略好于SUM最好**。


# PosPool: An Extremely Simple Local Aggregation Operator

直接使用相对位移和分块特征相乘。

$$
\begin{equation}
G\left(\Delta \mathbf{p}_{i j}, \mathbf{f}_{j}\right)=\text { Concat }\left[\Delta x_{i j} \mathbf{f}_{j}^{0} ; \Delta y_{i j} \mathbf{f}_{j}^{1} ; \Delta z_{i j} \mathbf{f}_{j}^{2}\right]
\end{equation}
$$

一种更复杂的是用一个6维向量拓展到和feature一个维度，再相乘
$$
\begin{equation}
\begin{aligned}
\mathcal{E}^{m}(x, y, z)=&\left[\operatorname { s i n } \left(100 x / 1000^{6 m / d}, \cos \left(100 x / 1000^{6 m / d}\right),\right.\right.\\
& \sin \left(100 y / 1000^{6 m / d}, \cos \left(100 y / 1000^{6 m / d}\right)\right.\\
& \sin \left(100 z / 1000^{6 m / d}, \cos \left(100 z / 1000^{6 m / d}\right)\right] .
\end{aligned}
\end{equation}
$$

$$
\begin{equation}
G\left(\Delta \mathbf{p}_{i j}, \mathbf{f}_{j}\right)=\mathcal{E} \odot \mathbf{f}_{i j}
\end{equation}
$$

# Experiments
![](/assets/img/20210604/LAOF3.png)

结论：**the PosPool operators achieve top or close-to-top performances using varying network hyper-parameters on all datasets, showing its strong stability and adaptability.**