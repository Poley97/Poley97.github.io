---
layout: post
title: 'Hidden Attribute Matters: Transferring Compositive Attribute via
Semi-Supervised Learning for High-Performance Object Re-identification'
date: 2021-05-31
author: Poley
cover: '/assets/img/20210531/HAM.png'
tags: 论文阅读
---

# Introduction

人类使用物体的不同属性作为识别物体的基本信息，同样最近的工作表明利用属性可以提升re-ID的性能。但是一方面，标注属性是很费时间的，只有很少几个数据集有属性标注。另一方面，不同数据集之间区别很大，一个数据集的属性分类在另一个上不一定好用。所以对于只有简单ID标注的数据集，更多的方法还是关注于悬系image本身的特征表示。

一般使用一个CNN来提取一个属性的特征，所以当多属性的是偶，问题变的很复杂。本文将原始属性合成一种高维属性，来指导CNN在高维空间中学习分布。之后提出 **Graph Enhancement Transferable Network(GETN)** 来提取隐式的合成特征，对于那些只有ID标注的数据集。

对于没有标注的数据集，提出了**Graph Enhancement Clustering(GEC)**来提取**伪复合属性(pseudo compositive attributive labels)**。和其余需要先验类别数的聚类方法不同，GEC通过构建图，利用邻域信息和节点关系来自动的几何positive nodes来找到最近接的类别数k。

# Related works

略

# Proposaed Approach

## Overview

一般当从一个数据集转换到另一个数据集上的时候，由于数据分布的变化（由于光照、相机位置、遮挡等），会导致显著的domain gap。这会导致网络直接用于另一个数据集上时性能的显著下降。本文提出的结构对domain gap有很好的适应能力，通过应用伪合成属性标注。

## Graph Enhancement Clustering(GEC)

图通过特征之间的k-NN来建立，其亲和度矩阵如下

$$
\begin{equation}
A_{i j}=1-\frac{x_{i}^{T}}{\left\|x_{i}\right\|_{2}} \frac{x_{j}}{\left\|x_{j}\right\|_{2}}
\end{equation}
$$

一个节点是一个sample的特征，图的建立是在pre-train网络提取了所有sample的特征之后进行的。然后通过融合图中每个节点的top-k近邻来更新节点信息。经过若干次更新得到聚类结果。 注意，这个图是没有参数的，因为它由KNN直接建立。之后使用图中聚类提取的特征来和原始特征进行融合,作为一种channel的权重，如下所示

$$
\begin{equation}
\begin{array}{c}
x_{i}^{(l+1)}=(1-l r) x_{i}^{(l)}+\operatorname{lr} \Delta x_{i}^{(l)} \\
\Delta x_{i}^{(l)}=\mathcal{F}_{j \in \mathfrak{N}_{i}}\left(x_{j}^{(l)}\right)
\end{array}
\end{equation}
$$

$$
\begin{equation}
\begin{array}{c}
S^{k^{(l)}}=A^{(l)} \odot \mathcal{A}^{k^{(l)}} \\
\mathcal{F}_{j \in \mathfrak{N}_{i}}\left(x_{j}^{(l)}\right)=\sum_{j \in \mathfrak{N}_{i}} \widetilde{S_{i j}^{k}}^{(l)} x_{j}^{(l)}
\end{array}
\end{equation}
$$

$$
\begin{equation}
C E\left(Z, Z^{\prime}\right)=\frac{Z^{\prime}}{\left\|Z^{\prime}\right\|_{2}} \odot Z+Z^{\prime}
\end{equation}
$$
## Generate pseudo compositive attribute label

虽然GEC可以有效的提取特征，从预训练模型中，但是这个特征不能直接的用于监督模型的学习。但是使用labels来训练模型是最好的监督模型来学习有辨识度的特征的犯法，所以提出一种时废来产生pseudo labels for nodes in the graph。

通过上述图的聚类，属于同一属性的特征会聚的很近，可以划分为一个子图。所以不同的子图划分为不同的**hidden compositive attribute labels**，这样，如何划分子图，也就是一个无偏的阈值，来filtering the length of edges 划分的关键。

对一个点的近邻亲和度排序
$$
\begin{equation}
\begin{array}{c}
A_{i}^{\prime}:=\left\{A_{i, j}^{\prime} \mid A_{i, j+1}^{\prime} \geq A_{i, j}^{\prime}, \forall A_{i, j}^{\prime} \in A_{i}, j \in \mathfrak{N}_{i}\right\}, \\
t=\left\{\begin{array}{lr}
\max _{i \in\{1 \ldots n\}} A_{i, 3}^{\prime}, & k>3 \\
\max _{i \in\{1 \ldots n\}} A_{i, k-1}^{\prime}, & 1<k \leq 3
\end{array}\right.
\end{array}
\end{equation}
$$
决定了阈值$t$，之后得到子图

由于会有一些节点的最短边都不符合阈值，使他们孤立。这样他们会变成一个单独的pseudo compositive attribute labels，因此将他们归纳与离得最近的节点的子图。

![](/assets/img/20210531/HAMF4.png)

## Discussion
K的选择是需要注意的，太大了会导致很多的negative nodes会出现，也就是不该属于这个sub-graph的Node.如果太小了，会导致图G被划分为太多的子图，产生不正确的pseudo compositive attribute。

>注：这里的图是先通过pretrain网络提取所有samples的特征，每个samples一个特征，然后建立一个图。通过若干次更新完成聚类，再划分子图，形成pseudo labels。

# Experiments

略