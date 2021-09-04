---
layout: post
title: 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'
date: 2021-09-02
author: Poley
cover: '/assets/img/20210902/EfficientNet.png'
tags: 论文阅读
---

>参考博客: https://zhuanlan.zhihu.com/p/96773680

对网络的扩展可以通过增加网络层数（depth，比如从 ResNet (He et al.)从resnet18到resnet200 ）, 也可以通过增加宽度，比如WideResNet (Zagoruyko & Komodakis, 2016)和Mo-bileNets (Howard et al., 2017) 可以扩大网络的width (#channels), 还有就是更大的输入图像尺寸(resolution)也可以帮助提高精度。如下图所示: (a)是基本模型，（b）是增加宽度，（c）是增加深度，（d）是增大属兔图像分辨率，（d）是EfficientNet，它从三个维度均扩大了，但是扩大多少，就是通过作者提出来的**复合模型扩张方法**结合**神经结构搜索（NAS）**技术获得的。
![](/assets/img/20210902/EfficientNetF2.png)

网络模型可以建模如下
$$
\begin{equation}
\mathcal{N}=\mathcal{F}_{k} \odot \ldots \odot \mathcal{F}_{2} \odot \mathcal{F}_{1}\left(X_{1}\right)=\bigodot_{j=1 \ldots k} \mathcal{F}_{j}\left(X_{1}\right)
\end{equation}
$$

对于具有多个Stage的模块（即类似结构），上述表达可以简化为
$$
\begin{equation}
\mathcal{N}=\bigodot_{i=1 \ldots s} \mathcal{F}_{i}^{L_{i}}\left(X_{\left\langle H_{i}, W_{i}, C_{i}\right\rangle}\right)
\end{equation}
$$

主要参数为$d,w,r$分别对应网络的深度，宽度和分辨率。在最大化网络准确率的情况下，可以建模如下
$$
\begin{equation}
\begin{array}{ll}
\max _{d, w, r} & \operatorname{Accuracy}(\mathcal{N}(d, w, r)) \\
\text { s.t. } & \mathcal{N}(d, w, r)=\bigodot_{i=1 \ldots s} \hat{\mathcal{F}}_{i}^{d \cdot \hat{L}_{i}}\left(X_{\left\langle r \cdot \hat{H}_{i}, r \cdot \hat{W}_{i}, w \cdot \hat{C}_{i}\right\rangle}\right) \\
& \operatorname{Memory}(\mathcal{N}) \leq \text{targetmemory } \\
& \operatorname{FLOPS}(\mathcal{N}) \leq \text {targetflops}
\end{array}
\end{equation}
$$

之后作者提出了复合扩张方法，将问题转换为如下形式

depth: $d=\alpha^{\phi}$
width: $w=\beta^{\phi}$
resolution: $r=\gamma^{\phi}$
$$
\begin{aligned}
&\text { s.t. } \alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2 \\
&\qquad \alpha \geq 1, \beta \geq 1, \gamma \geq 1
\end{aligned}
$$

即本文的核心，其求解方式如下
固定公式中的$\phi=1$，然后通过网格搜索（grid search）得出最优的$\alpha,\beta,\gamma$，得出最基本的模型EfficientNet-B0.
固定$\alpha,\beta,\gamma$的值，使用不同的φ，得到EfficientNet-B1, ..., EfficientNet-B7
φ的大小对应着消耗资源的大小，相当于：

当$\phi=1$时，得出了一个最小的最优基础模型；
增大$\phi$时，相当于对基模型三个维度同时扩展，模型变大，性能也会提升，资源消耗也变大。
对于神经网络搜索，作者使用了和 MnasNet: Platform-awareneural architecture search for mobile 一样的搜索空间和优化目标。