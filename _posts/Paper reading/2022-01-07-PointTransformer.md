---
layout: post
title: 'Point Transformer'
date: 2022-1-7
author: Poley
cover: '/assets/img/20211228/PointTransformer.png'
tags: 论文阅读
---

这边文章和PCT基本于隔天发表，是Point Cloud里使用Transformer应用的比较经典的文章。本文的思路更类似于PointNet++的拓展。
Transformer本身的特性其实比较适合于点云，因为其具有几个特点：
+ 本质其是一种Set Operater
+ 顺序不变形
+ 需要位置编码来确保数据作为一个Set被处理的同时保留原有信息
  
这三点都和点云数据的特点非常匹配。因此，Transformer应该可以很自然的被利用到点云中。然而点云数量众多，对于Transformer $O(n^2)$的复杂度不太友好，因此一般选择在Local范围内进行self-attention。本文也是这个思路。

Transformer主要分两种形式
+ standard scalar dot-product attention layer：
$$
\begin{equation}
\mathbf{y}_{i}=\sum_{\mathbf{x}_{j} \in \mathcal{X}} \rho\left(\varphi\left(\mathbf{x}_{i}\right)^{\top} \psi\left(\mathbf{x}_{j}\right)+\delta\right) \alpha\left(\mathbf{x}_{j}\right)
\end{equation}
$$
+ vector attention,即使用非点乘的方式计算注意力矩阵，如下
$$
\begin{equation}
\mathbf{y}_{i}=\sum_{\mathbf{x}_{j} \in \mathcal{X}} \rho\left(\gamma\left(\beta\left(\varphi\left(\mathbf{x}_{i}\right), \psi\left(\mathbf{x}_{j}\right)\right)+\delta\right)\right) \odot \alpha\left(\mathbf{x}_{j}\right)
\end{equation}
$$

这里作者使用的是第二种，在其代码注释中，作者说这样的效果好过点积。同时，作者表明在点云处理中，位置编码具有非常重要的地位，因此和一般的attention不同，这里在Attention矩阵和Value上都加入了位置编码的信息，如下所示。
$$
\begin{equation}
\mathbf{y}_{i}=\sum_{\mathbf{x}_{j} \in \mathcal{X}(i)} \rho\left(\gamma\left(\varphi\left(\mathbf{x}_{i}\right)-\psi\left(\mathbf{x}_{j}\right)+\delta\right)\right) \odot\left(\alpha\left(\mathbf{x}_{j}\right)+\delta\right)
\end{equation}
$$

![](/assets/img/20211228/PointTransformerF2.png)

其整体结构如下所示
![](/assets/img/20211228/PointTransformerF3.png)
![](/assets/img/20211228/PointTransformerF4.png)

并不复杂，其中Transformer模块也类似于PointNet++的set abstraction模块，先用KNN graph进行近邻搜索的group，再在组内进行self-attention来有效的减小计算量。

其在ModleNet上达到了93.6的性能，达到了当时的SOTA（当时超过93就可以算SOTA基本上）
![](/assets/img/20211228/PointTransformerT3.png)