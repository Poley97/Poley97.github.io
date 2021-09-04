---
layout: post
title: 'Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks'
date: 2021-08-24
author: Poley
cover: '/assets/img/20210824/ExternalAtt.png'
tags: 论文阅读
---

>参考博客：https://0809zheng.github.io/2021/08/09/external.html

self-attention具有$O(n^2)$的复杂度，并且忽略了不同样本之间的关系。本文提出External attention 隐式的包含了全部数据之间的correlations。并且具有线性复杂度。在多个cv tasks上具有不错的效果。

本文使用两个memory来代替self attention中的keys和values，这两者对于单独的样本是独立的，并且在整个数据集内共享，因此起到了一个很强的**正则作用，并提成了attention机制的泛化能力**。

通过这种方式，可以将算法复杂度变为线性。在实践中，两个memory模块通过两个线性层来实现，使得整个网络变为**all-MLP architecture named EAMLP**

其和Self-attention的对比如下图所示

![](/assets/img/20210824/ExternalAttF1.png)

由于本文的memory通过MLP实现，因此可以通过BP算法端到端实现，而不需要额外的迭代算法。

传统的self attention 如下所示

$$
\begin{equation}
\begin{aligned}
A &=(\alpha)_{i, j}=\operatorname{softmax}\left(Q K^{T}\right) \\
F_{o u t} &=A V
\end{aligned}
\end{equation}
$$

其中$Q \in \mathbb{R}^{N \times d^{\prime}}$，$K \in \mathbb{R}^{N \times d^{\prime}}$，$V \in \mathbb{R}^{N \times d^{\prime}}$

一种简化的方法是直接通过输入特征计算attention map，如下
$$
\begin{equation}
\begin{aligned}
A &=\operatorname{softmax}\left(F F^{T}\right) \\
F_{\text {out }} &=A F .
\end{aligned}
\end{equation}
$$

但是尽管简化，这还是$O(n^2)$的复杂度。因此对于图像来说，逐像素的self-attention成本太高，只能分为若干个patch处理。

Self-attention可以是做事使用自己值的线性组合来对输出特征进行refine,但是这一定需要$N\times N$大小的矩阵么？因此作者引入extenal memory unit $M \in \mathbb{R}^{S \times d}$

$$
\begin{equation}
\begin{aligned}
A &=(\alpha)_{i, j}=\operatorname{Norm}\left(F M^{T}\right), \\
F_{\text {out }} &=A M
\end{aligned}
\end{equation}
$$

时间中，M被分为两个不同的memory unit如下
$$
\begin{equation}
\begin{aligned}
A &=(\alpha)_{i, j}=\operatorname{Norm}\left(F M_k^{T}\right), \\
F_{\text {out }} &=A M_v
\end{aligned}
\end{equation}
$$

其伪代码如下

![](/assets/img/20210824/ExternalAttA1.png)

由于attention是通过矩阵乘法而不是余弦距离得到的，因此对于输入特征的scale非常敏感。为了解决这个问题，作者使用了double-normalization,其独立的对行和列进行归一化，如下所示

$$
\begin{equation}
\begin{aligned}
(\tilde{\alpha})_{i, j} &=F M_{k}^{T} \\
\hat{\alpha}_{i, j} &=\exp \left(\tilde{\alpha}_{i, j}\right) / \sum_{k} \exp \left(\tilde{\alpha}_{k, j}\right) \\
\alpha_{i, j} &=\hat{\alpha}_{i, j} / \sum_{k} \hat{\alpha}_{i, k}
\end{aligned}
\end{equation}
$$

同样，和self-attention一样，external attention可以引入 multi-head 机制，如下图所示。

![](/assets/img/20210824/ExternalAttF2.png)

![](/assets/img/20210824/ExternalAttA2.png)

值得注意的是，其实验中展现出其在点云分类上也有不错的效果。笔者之后打算进一步尝试一下。