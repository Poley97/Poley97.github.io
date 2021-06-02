---
layout: post
title: 'Unsupervised Feature Learning via Non-Parametric Instance Discrimination'
date: 2021-06-01
author: Poley
cover: '/assets/img/20210601/UFLNPID.png'
tags: 论文阅读
---
> 论文链接： https://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.pdf
> 参考博客 ： https://blog.csdn.net/qq_16936725/article/details/51147767

# Introduction

本文提出的无监督学习的新方法来自于物体识别领域的几个发现。在ImageNet上，top-5错误率远低于top-1错误率，并且置信度第二高的类别往往和图像本身具有较高的视觉关联。如下图所示

![](/assets/img/20210601/UFLNPIDF1.png)

这样的发现说明discriminative learning method可以自动的发现语义类别中显著的相似性，而不需要特别的指引。**也就是说，这种显著的相似性不是从语义标注中学到的，而是来自于视觉数据本身**。

因此作者把class-wise的监督用到instance-wise的监督上，是否可以通过discriminative learning学习到一种Metric，来反应instance之间显著的相似性？每个image都以他们自己的形式被distinct，每个都显著的和同语义类别的其他images不一样。如果我们学会如何区分两个独立的instance，而不用任何语义类别，**可能就可以得到一种捕捉instance间显著相似性的表达，就像class-wise捕捉类间相似性一样。**。由于这个本身也是一个instance-level discrimination 的任务，所以也可以从最新的discriminative supervised learning中获得好处，比如新的网络结构。

但是，我们也遇到了一些问题，比如**classes数量**数量的激增。如果一个实例是一个类别，那么ImageNet可能有1.2M个类别而不是100类，因此softmax显得不再可行。因此使用一种方法来模拟softmax的分布，通过**noise-contrastive estimation(NCE)**,并利用一个**proximal regularization**来稳定学习过程。

在训练和测试过程中，使用了**non-parametric approach**，将**instance-level discrimination**问题变为一个**metirc learning problem**。两个实例之间的距离直接使用non-parameteic way来实现。

# Related works
略

# Approach
![](/assets/img/20210601/UFLNPIDF2.png)

目的是学习一个embedding function，而不用监督。其距离衡量如下
$$
\begin{equation}
d_{\boldsymbol{\theta}}(x, y)=\left\|f_{\boldsymbol{\theta}}(x)-f_{\boldsymbol{\theta}}(y)\right\|
\end{equation}
$$

## Non-Parametric Softmax Classifier

### Parametric Classifier

$$
\begin{equation}
P(i \mid \mathbf{v})=\frac{\exp \left(\mathbf{w}_{i}^{T} \mathbf{v}\right)}{\sum_{j=1}^{n} \exp \left(\mathbf{w}_{j}^{T} \mathbf{v}\right)}
\end{equation}
$$
其中$v$是网络输出的特征，$w$是类别的权重向量。如果特征维度是128，一共120M个实例（类别），则这一层的参数超过15亿，这是不可接受的。

### Non-Parametric Classifier

$$
\begin{equation}
P(i \mid \mathbf{v})=\frac{\exp \left(\mathbf{v}_{i}^{T} \mathbf{v} / \tau\right)}{\sum_{j=1}^{n} \exp \left(\mathbf{v}_{j}^{T} \mathbf{v} / \tau\right)}
\end{equation}
$$

**其中$\tau$是温度参数，控制了分布的concentration(集中度?)**，它对于监督学习很重要，在这里，同样对**tuning the concentration of v on our unit sphere**很必要。

学习目标是最大化联合概率密度，也就是最小化负对数似然
$$
\begin{equation}
J(\boldsymbol{\theta})=-\sum_{i=1}^{n} \log P\left(i \mid f_{\boldsymbol{\theta}}\left(x_{i}\right)\right)
\end{equation}
$$

### Learning with A Memory Bank

为了计算上述的条件概率，需要${v_i}$ for all the images。为了避免每次都计算产生的巨大计算量，所以需要维护一个feature memory bank V 来存储所有图片的特征。字典V里的所有表达用随机的向量来初始化。当一个新的特征进入了字典即完成了更新。

## Noise-Contrastive Estimation

使用NCE来模拟full=softmax。但是避免计算全部instance的相似度，将这个问题转换为一个二分类问题。其基本思想是将多分类任务转化为一系列二分类任务，二分类任务是判断样本是来自与真实数据还是噪声数据。特别的，我们将概率改写为
$$
\begin{equation}
P(i \mid \mathbf{v})=\frac{\exp \left(\mathbf{v}^{T} \mathbf{f}_{i} / \tau\right)}{Z_{i}} 
\end{equation}
$$
$$
\begin{equation}
Z_{i}=\sum_{j=1}^{n} \exp \left(\mathbf{v}_{j}^{T} \mathbf{f}_{i} / \tau\right)
\end{equation}
$$

同时划定噪声服从均匀分布$$P_{n}=1 / n$$

假设噪声m倍多余data samples，则sample $i$和feature$\bold{v}$来自于data distribution的后延概率为

$$
\begin{equation}
h(i, \mathbf{v}):=P(D=1 \mid i, \mathbf{v})=\frac{P(i \mid \mathbf{v})}{P(i \mid \mathbf{v})+m P_{n}(i)}
\end{equation}
$$

对这个后验概率进行优化得到
$$
\begin{equation}
\begin{aligned}
J_{N C E}(\boldsymbol{\theta}) &=-E_{P_{d}}[\log h(i, \mathbf{v})] \\
&-m \cdot E_{P_{n}}\left[\log \left(1-h\left(i, \mathbf{v}^{\prime}\right)\right)\right] .
\end{aligned}
\end{equation}
$$

由于上述的分母计算量很大，所以将其视为一个常数，使用Monte Carlo approximation。
$$
\begin{equation}
Z \simeq Z_{i} \simeq n E_{j}\left[\exp \left(\mathbf{v}_{j}^{T} \mathbf{f}_{i} / \tau\right)\right]=\frac{n}{m} \sum_{k=1}^{m} \exp \left(\mathbf{v}_{j_{k}}^{T} \mathbf{f}_{i} / \tau\right)
\end{equation}
$$

## Proximal Regularization
![](/assets/img/20210601/UFLNPIDF3.png)

由于所有的类别（一个instance就是一个类别）在每个epoch只遍历一次，所以学习过程可能会由于采样波动而产生震荡，所以加入一个正则项，来鼓励光滑的学习曲线，如下。
$$
\begin{equation}
-\log h\left(i, \mathbf{v}_{i}^{(t-1)}\right)+\lambda\left\|\mathbf{v}_{i}^{(t)}-\mathbf{v}_{i}^{(t-1)}\right\|_{2}^{2}
\end{equation}
$$

这样最终的损失变为

$$
\begin{equation}
\begin{aligned}
J_{N C E}(\boldsymbol{\theta}) &=-E_{P_{d}}\left[\log h\left(i, \mathbf{v}_{i}^{(t-1)}\right)-\lambda\left\|\mathbf{v}_{i}^{(t)}-\mathbf{v}_{i}^{(t-1)}\right\|_{2}^{2}\right] \\
&-m \cdot E_{P_{n}}\left[\log \left(1-h\left(i, \mathbf{v}^{\prime(t-1)}\right)\right)\right]
\end{aligned}
\end{equation}
$$

## Weighted k-Nearest Neighbor Classifier

使用KNN作为最后分类的依据，而不是SVM，实验表明KNN具有更好的性能。