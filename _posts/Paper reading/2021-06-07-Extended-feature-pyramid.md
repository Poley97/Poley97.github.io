---
layout: post
title: 'Extended Feature Pyramid Network for Small Object Detection'
date: 2021-06-07
author: Poley
cover: '/assets/img/20210607/EFP.png'
tags: 论文阅读
---

> 论文链接 ： https://arxiv.org/pdf/2003.07021v1.pdf

# Introduction

![](/assets/img/20210607/EFPF1.png)

尽管FPN提升了multi-scale detection的表现，但是这种启发式的mapping机制可能会混淆小目标检测，如上图所示，小目标和中目标可能被迫共享一个feature map，而大目标则可以选择自己最适合的level。这在AP上也可以反映出来，小目标的AP在这种情况下远低于大目标的。

一种自然的想法就是提高feature的分辨率，比如一些超分辨率方法。一种方法是直接对输入图像做超分辨率，但是这样计算成本很高；另一种方法是FPN底层特征做超分辨率，但是这样SR方法经常会想象出一些不存在的特征，也就是倾向于虚构一些假的特征，最终导致false positives。

本文提出**extended feature pyramid network(EFPN)**，使用一种大尺度的SR特征with充分的细节来将small 和 medium 目标检测解耦。同时这个高分辨率feature map依赖于CNN产生的原始真是特征，而不是其他方法中的凭空想象。

# Our Approach


## Extended Feature Pyramid Network
在原有FPN的基础上，通过SR得到一个更高分辨率并且可靠的特征。

![](/assets/img/20210607/EFPF2.png)

如下所示，$P_2$和$P_3$特征结合，产生$P_3'$，在经过$C_2'$的融合，得到$P_2'$也就是可靠的高分辨率特征。其中，$C_2'$和$C_2$的区别如下所示

![](/assets/img/20210607/EFPT1.png)

$P_2'$通过求和产生
$$
\begin{equation}
P_{2}^{\prime}=P_{3}^{\prime} \uparrow_{2 \times}+C_{2}^{\prime}
\end{equation}
$$

特征层的分配如下
$$
\begin{equation}
l=\left\lfloor l_{0}+\log _{2}(\sqrt{w h} / 224)\right\rfloor
\end{equation}
$$

## Feature Texture Transfer

FFT模块用于进行超分辨率feature和提取regional textures from reference fetaures simultaneously。FFT可以阻止$P_2$的噪声直接传递到额外的金字塔层，导致淹没了有用的语义信息。其结合了高层的语音信息和底层的关键的Local deteails但是消除了一些噪声。结构如下图所示。

![](/assets/img/20210607/EFPF3.png)

可以表示为

$$
\begin{equation}
P_{3}^{\prime}=\mathbf{E}_{\mathbf{t}}\left(P_{2} \| \mathbf{E}_{\mathbf{c}}\left(P_{3}\right) \uparrow_{2 \times}\right)+\mathbf{E}_{\mathbf{c}}\left(P_{3}\right) \uparrow_{2 x}
\end{equation}
$$

Sub-pixel convolution通过转移channel来增强hw维度的特征，如下所示。
$$
\begin{equation}
\mathbf{P S}(F)_{x, y, c}=F_{\lfloor x / r\rfloor,\lfloor y / r\rfloor, C \cdot r \cdot \bmod (y, r)+C \cdot \bmod (x, r)+c}
\end{equation}
$$

## Cross Resolution Distillation

高分辨率的输入往往具有更好的检测效果，但是分辨率高到一定程度之后，性能便不再提升，如下所示。

![](/assets/img/20210607/EFPF5.png)

因此，本文提出一个cross resolution distillation ，将高分辨率的特征作为监督，如下所示。

![](/assets/img/20210607/EFPF4.png)

Loss计算如下
$$
\begin{equation}
L=L_{f b b}\left(P_{3}^{\prime}, P_{3}^{2 \times}\right)+L_{f b b}\left(P_{2}^{\prime}, P_{2}^{2 \times}\right)
\end{equation}
$$

其中fbb是本文提出的**foreground-background-balanced loss**，用于解决前景和背景点之间面积不平衡的问题，提高EFPN的综合质量。其中包含两部分：

1. **Global reconstruction loss**:
$$
\begin{equation}
L_{g l o b}\left(F, F^{t}\right)=\left\|F^{t}-F\right\|_{1}
\end{equation}
$$
2. **Positive patch loss**:
$$
\begin{equation}
L_{p o s}\left(F, F^{t}\right)=\frac{1}{N} \sum_{(x, y) \in P_{p o s}}\left\|F_{x, y}^{t}-F_{x, y}\right\|_{1}
\end{equation}
$$

最后的综合loss

$$
\begin{equation}
L_{f b b}\left(F, F^{t}\right)=L_{g l o b}\left(F, F^{t}\right)+\lambda L_{p o s}\left(F, F^{t}\right)
\end{equation}
$$

个人理解相当于数倍突出正样本像素的重构loss。

# Experiments

![](/assets/img/20210607/EFPT2.png)

![](/assets/img/20210607/EFPT3.png)