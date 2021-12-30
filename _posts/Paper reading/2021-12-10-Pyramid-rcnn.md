---
layout: post
title: 'Pyramid R-CNN:
Towards Better Performance and Adaptability for 3D Object Detection'
date: 2021-12-10
author: Poley
cover: '/assets/img/20211206/PyramidRCNN.png'
tags: 论文阅读
---

## Method
算法的整体结构如下所示
![](/assets/img/20211206/PyramidRCNNF2.png)

本方法的一阶段特征提取网络可以是任意网络，只需要提供两个信息：3D proposal和 corresponding Points of Interest。之后通过三个模块来提取特征：
+ RoI grid Pyramid：使用多个不同大小的Enlarge Proposal和其中对应的grid points来提取特征；
+ Density-Aware Radius Prediction: 每个grid points的focusing radius $r$ 通过本网络预测得到；
+ RoI-grid Attention: 被focusing radius $r$ 参数化，来聚合Points of Interest 特征。

最后，送入两个独立的回归和分类head。完成预测。

### RoI-grid Pyramid

给定一个目标框大小，以其为中心坐标得到的相对grid坐标点如下所示
$$
\begin{equation}
p_{\text {grid }}^{i j k}=\left(\frac{W}{N_{w}}, \frac{L}{N_{l}}, \frac{H}{N_{h}}\right) \cdot(0.5+(i, j, k))+\left(x_{c}, y_{c}, z_{c}\right),
\end{equation}
$$
这里Pyramid使用多个大小不同的框以及其中对应的grid points中来聚合特征，即将原来的proposal放大一个倍数$\rho$，得到新的grid points 如下
$$
\begin{equation}
p_{g r i d}^{i j k}=\left(\frac{\rho_{w} W}{N_{w}^{\prime}}, \frac{\rho_{l} L}{N_{l}^{\prime}}, \frac{\rho_{h} H}{N_{h}^{\prime}}\right) \cdot(0.5+(i, j, k))+\left(x_{c}, y_{c}, z_{c}\right)
\end{equation}
$$

### RoI-grid Attention

作者总结了RoI-grid Arregation中主要有以下这么几种操作

+ Preliminary: 最基本的形式，一般使用最大池化
$$
\begin{equation}
f_{g r i d}^{p o o l}=\operatorname{maxpool}_{i \in \Omega(r)}\left(V^{i}\right)
\end{equation}
$$
+ Graph-based Operators: 使用相对位移（边）的embedding的权重，如下所示
$$
\begin{equation}
f_{\text {grid }}^{\text {graph }}=\sum_{i \in \Omega(r)} W\left(Q_{\text {pos }}^{i}\right) \odot V^{i},
\end{equation}
$$
其中
$$
\begin{equation}
Q_{\text {pos }}^{i}=\operatorname{Linear}\left(p_{i}-p_{\text {grid }}\right)
\end{equation}
$$

+ Attention-based Operators: 使用Attention或者Transformer来进行特征融合，如下所示
$$
\begin{equation}
f_{\text {grid }}^{\text {atten }}=\sum_{i \in \Omega(r)} W\left(Q_{\text {pos }}^{i} K^{i}\right) \odot V^{i}
\end{equation}
$$
$$
\begin{equation}
f_{g r i d}^{t r}=\sum_{i \in \Omega(r)} W\left(K^{i}+Q_{p o s}^{i}\right) \odot\left(V^{i}+Q_{p o s}^{i}\right)
\end{equation}
$$
其中
$$
\begin{equation}
K^{i}=\operatorname{Linear}\left(f_{i}\right)
\end{equation}
$$

+ RoI-grid Attention: 这里作者想要融合上述几种聚合方法，进行自适应的加权。将上述几个式子整合到一个式子中得到
$$
\begin{equation}
\begin{gathered}
f_{\text {grid }}=\sum_{i \in \Omega(r)} W\left(\sigma_{k} K^{i}+\sigma_{q} Q_{\text {pos }}^{i}+\sigma_{q k} Q_{p o s}^{i} K^{i}\right) \\
\odot\left(V^{i}+\sigma_{v} Q_{p o s}^{i}\right)
\end{gathered}
\end{equation}
$$
其中各个权重都是学习参数，得以自适应改变权重，得到更灵活的特征聚合方式。
![](/assets/img/20211206/PyramidRCNNF4.png)

### Density-Aware Radius Prediction
使用可变的半径来聚合特征，最大的问题是不可导性，作者在这里通过一些策略，使得对于动态半径的估计变得可导，个人认为是本文最主要的创新点。

传统的固定半径的近邻搜索以概率形式可以表示如下
$$
\begin{equation}
p(i \mid r)= \begin{cases}0 & \left\|p_{i}-p_{\text {grid }}\right\|_{2}>r \\ 1 & \left\|p_{i}-p_{\text {grid }}\right\|_{2} \leq r\end{cases}
\end{equation}
$$
最终得到的融合特征可以视为以下期望
$$
\begin{equation}
f_{g r i d}=\mathbb{E}_{i \sim p(i \mid r)}\left[W^{i} \odot V^{i}\right],
\end{equation}
$$

这里提出一个新的概率分布来代替上述概率分布，其需要具有和上述概率分布类似的特性，并且包含少数$r$外的点来感知附近的环境。作者在这里建模这个概率密度函数如下
$$
\begin{equation}
s(i \mid r)=1-\operatorname{sigmoid}\left(\frac{\left\|p_{i}-p_{\text {grid }}\right\|_{2}-r}{\tau}\right),
\end{equation}
$$

那么此时聚合特征对于半径$r$的导数变为如下
$$
\begin{equation}
\nabla_{r} f_{\text {grid }}=\nabla_{r} \mathbb{E}_{i \sim s(i \mid r)}\left[W^{i} \odot V^{i}\right]
\end{equation}
$$

注意到此时$r$仍然是hi不可行的，因为其作为概率分布的参数不可导，这里引入了重参数化的trick，将分布参数转移到求期望的过程中，故梯度可以变为如下可导形式

$$
\begin{equation}
\nabla_{r} f_{\text {grid }}=\mathbb{E}_{i \sim U(\epsilon)}\left[\nabla_{r}\left[s(i, r) \cdot W^{i} \odot V^{i}\right]\right]
\end{equation}
$$

其中$U(\epsilon)=1$代表全空间采样，实际应用中，全空间采样会有很多概率趋近于0，没有意义，故这里为了节省计算量，限制了一下计算的采样点范围，比如$r+5\tau$，最终得到的动态半径如下图所示

![](/assets/img/20211206/PyramidRCNNF5.png)

## Experiments
![](/assets/img/20211206/PyramidRCNNT1T2.png)
