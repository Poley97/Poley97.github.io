---
layout: post
title: 'SPARSE DETR: EFFICIENT END-TO-END OBJECT DETECTION WITH LEARNABLE SPARSITY'
date: 2023-3-7
author: Poley
cover: '/assets/img/20230227/SparseDETR.jpg'
tags: 论文阅读  
---

本文作者提出，实验表明，在已经训练完成的DETR中，去除大部分背景token对于模型的性能基本没有影响。受此启发，本文提出sparse DETR, 主要目的是去除DETR模型中冗余的token，实现收敛和推理速度的提升。

常用的两种DETR，原始版本，复杂度N^2，deformable，复杂度N^k。这使得deformable可以引入多尺度，但是这样也使其拥有20倍的token，其计算速度仍然是一个瓶颈。

本文实际上相当于在deformable上的进一步改进。这里的sparse主要针对encoder。只更新在之后会被ref的encoder token，从而减小encoder中selfattention的计算量。这里相当于进一步简化deformable，deformable只在key维度做了简化，这里进一步在query维度上也稀疏化了。如下图所示

![](/assets/img/20230227/SparseDETRF1.jpg)

## ENCODER TOKEN SPARSIFICATION

这部分的主要是指定encoder中的token的稀疏化，不对无用的token进行处理。这里设定了一个$\rho$作为保留token的比例，则在encoder中,网络对token的处理可以表示为
$$
\begin{equation}
\mathbf{x}_i^j= \begin{cases}\mathbf{x}_{i-1}^j & j \notin \Omega_s^\rho \\ \operatorname{LN}\left(\operatorname{FFN}\left(\mathbf{z}_i^j\right)+\mathbf{z}_i^j\right) & j \in \Omega_s^\rho, \text { where } \mathbf{z}_i^j=\operatorname{LN}\left(\operatorname{Def} \operatorname{Attn}\left(\mathbf{x}_{i-1}^j, \mathbf{x}_{i-1}\right)+\mathbf{x}_{i-1}^j\right),\end{cases}
\end{equation}
$$

即encoder中选择性更新token。其中更新的比例是固定的。注意到，这里即使是不更新的token，仍然送入了encoder中作为key，使得显著的前景token仍然可以使用他们来更新自己。
此时，复杂度进一步从deformable的NK降低为SK。

## FINDING SALIENT ENCODER TOKENS

上述带来的直观问题就是，如何选择显著的token保留。本文提出使用transformer decode中的cross attention map来判断encoder中哪些token的saliency。这里首先介绍了通过objectness score分割来筛选前景方法的局限性。

这里作者认为使用objectness得分的不足在于，其虽然可以有效筛选，但是没有考虑到decoder的需求。
因此，这里创新性的提出，使用cross attention的attention map来进行监督。

这里的响应通过（Decoder cross-
Attention Map）DAM获得。dense的直接通过累加各layer的响应获得，deformable的相对麻烦，需要根据offset映射到对应的位置。

总体结构的示意图如下所示

![](/assets/img/20230227/SparseDETRF2.jpg)

其中，对于分类得分的监督使用二值交叉熵，如下所示

$$
\begin{equation}
\mathcal{L}_{\text {dam }}=-\frac{1}{N} \sum_{i=1}^N \operatorname{BCE}\left(g\left(\mathbf{x}_{\text {feat }}\right)_i, \operatorname{DAM}_i^{\mathrm{bin}}\right)
\end{equation}
$$

一般情况下，在训练中进行严重的剪枝会造成训练的不稳定，尤其是在训练的早期。但是这里作者指出，稀疏的特征不会影响训练，即使是训练早期。

## ADDITIONAL COMPONENTS

作者发现使用一个辅助网络在seleted token上进行检测（使用匈牙利匹配做target assign）有助于网络收敛的稳定，缓解了梯度的消失，甚至有助于提高检测性能。

最后，网络的整体结构如下所示
![](/assets/img/20230227/SparseDETRF3.jpg)

在网络的性能和收敛速度上，在当时均达到了比较优秀的水平，如下所示

![](/assets/img/20230227/SparseDETRT1.jpg)