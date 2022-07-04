---
layout: post
title: 'Group-Free 3D Object Detection via Transformers'
date: 2022-1-11
author: Poley
cover: '/assets/img/20220111/Groupfree.png'
tags: 论文阅读
---

本文提出一种无需启发式分组或聚类的基于点的3D目标检测方法。

基于点聚类（或者叫group）的目标检测方法，会因为had-crafted的grouping方法不能准确将点分配到齐对应的目标而造成性能的损失。本文就是解决这个问题，不用group，即group-free。但是笔者注意到，本文的研究开展在室内场景下，室内场景存在目标的重叠情况，如下图所示。而以自动驾驶场景位代表的室外激光雷达点云则不存在这个问题。不过，本文的思路仍然值得学习。因为笔者认为，通过attention自适应的聚合点云中的特征非常有意义。

![](/assets/img/20220111/GroupfreeF1.png)

在上图中，如果按照一般通过目标框来对点进行assign的方式，那么大目标沙发中就的真样本点就会包含不属于他的点，也就是图中的茶几。这样的错误分配会导致在后续的vote或者类似操作中的预测结果偏移（茶几正样本点的中心预测朝着自己的中心，因此在Vote的时候沙发的中心点会被带偏）。

为了避免这个问题，本文提出使用所有的点特征来计算每个object candidate的特征。这很自然的需要自适应的对不同的点特征进行加权，进而需要使用attention工具。

这里直接引入了Transformer的标准形式，同时在pos emb上进行了改进，使用前一个stage产生的中心预测作为下一个stage中Object candidate的位置编码输入。

整体的网络结构如下所示

![](/assets/img/20220111/GroupfreeF2.png)

笔者认为这样的结构是可以预料的。注意到，这里实际上进行了两次下采样。分别是输入之前进行的整体点云下采样和object candidate进行的下采样。对于使用Transformer的点云网络来说，这似乎是必须的。考虑到Transformer具有$O(n^2)$的复杂度，过高的点数是不可接受的。

在Backbone上，本文比较简单，使用PointNet++来实现。在Object Candidate Sampling上，作者给出了一点小创新，使用可选的三种方法，分别是
+ FPS：最远点采样，略；
+ k-Closest Points Sampling（KPS）：对点预测一个分数，按照分数采样。当点时距离物体中心最近的k个点之一的时候其是正样本。
+  k-Closest Points Sampling with NMS: 再上面的基础上再加一个中心预测，然后对于中心靠的过近的点做一个抑制。

这里使用的Attention是标准的形式，如下
$$
\begin{equation}
\operatorname{Att}\left(\mathbf{q}_{i},\left\{\mathbf{p}_{k}\right\}\right)=\sum_{h=1}^{H} W_{h}\left(\sum_{k=1}^{K} A_{i, k}^{h} \cdot V_{h} \mathbf{p}_{k}\right)
\end{equation}
$$
$$
\begin{equation}
A_{i, k}^{h}=\frac{\exp \left[\left(Q_{h} \mathbf{q}_{i}\right)^{T}\left(U_{h} \mathbf{p}_{k}\right)\right]}{\sum_{k=1}^{K} \exp \left[\left(Q_{h} \mathbf{q}_{i}\right)^{T}\left(U_{h} \mathbf{p}_{k}\right)\right]}
\end{equation}
$$

在上述形式下，其使用了标准的Self-attention和类似于DETR中加入可学习query变量的Cross-attention，如下
$$
\begin{equation}
\operatorname{Self}-\operatorname{Att}\left(\mathbf{o}_{i}^{(l)},\left\{\mathbf{o}_{j}^{(l)}\right\}\right)=\operatorname{Att}\left(\mathbf{o}_{i}^{(l)},\left\{\mathbf{o}_{j}^{(l)}\right\}\right)
\end{equation}
$$
$$
\begin{equation}
\operatorname{Cross}-\operatorname{Att}\left(\mathbf{o}_{i}^{(l)},\left\{\mathbf{z}_{j}^{(l)}\right\}\right)=\operatorname{Att}\left(\mathbf{o}_{i}^{(l)},\left\{\mathbf{z}_{j}^{(l)}\right\}\right)
\end{equation}
$$

总的来说创新并不多，只是单纯的解决启发式的人工设计group不准确的问题。实验结果如下所示
![](/assets/img/20220111/GroupfreeT1T2.png)

**其和DETR的不同在于，这个先Object candidate 做self attention再和全部点云做cross attention 的过程，相当于将Object candidate转换成了一个数据相关的query，而不像DETR那样直接使用独立的可学习变量进行没有额外监督的学习。这样可能会使得query包含更多更准确的信息。**另外一个和DETR的performance gap可能来自于Loss的不同，本文是手动assign的，而DETR是用set loss自动分配的，可能增加了学习的难度。

![](/assets/img/20220111/GroupfreeT10.png)
