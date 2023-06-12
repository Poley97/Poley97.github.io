---
layout: post
title: 'DEFORMABLE DETR: DEFORMABLE TRANSFORMERS
FOR END-TO-END OBJECT DETECTION'
date: 2022-07-11
author: Poley
cover: '/assets/img/20220711/DeformableDETR.jpg'
tags: 论文阅读
---


DETR的思路是将backbone输出特征图的像素展开成一维后当成了序列长度，而batch和channel的定义不变，同时添加位置编码。这样导致需要计算相关的点对很多。本文提出对于一个reference，只关注其周围的若干个点。

作者首先提出DETR存在的两个问题：1、训练收敛速度慢；2、对小目标检测能力差。而本文借鉴了deformable conv的思想，结合deformable卷积 attend to sparse spatial localtions以及DETR建模元素相关性的能力。提出Deformable DETR实现高效的特征提取。

![](/assets/img/20220711/DeformableDETRF1.jpg)

相比DETR，本文的改进有
+ 只关注query附近的部分位置，大大减小了计算复杂度
+ 同时关注多个scale的特征，代替了FPN实现了对多尺度信息的融合
+ 提出两阶段的DETR，使用proposal作为解码器中的object queries，体高精度

由于Transformer $O(N^2)$，在二维的图片上会进而变为$O(N^4)$的计算量，为算法带来较大的负担。这也使得DETR模型的训练较慢，而且对于小目标的精度较差（因为不能使用很高的特征图分辨率）。因此，需要使用一些方法来减小Transformer在CV应用上的计算量。目前来说，Transformer在CV上的应用可以分为三类，
+ 预定义稀疏注意力模式。将attention限制在一个local window里来减小计算量；
+ 数据驱动，根据query来关注不同的区域。即学习数据驱动的稀疏注意力；
+ 自注意力的低rank特性来简化计算。
![](/assets/img/20220711/DeformableDETRF2.jpg)

原本的注意力机制如下
$$
\begin{equation}
\operatorname{MultiHeadAttn}\left(\boldsymbol{z}_{q}, \boldsymbol{x}\right)=\sum_{m=1}^{M} \boldsymbol{W}_{m}\left[\sum_{k \in \Omega_{k}} A_{m q k} \cdot \boldsymbol{W}_{m}^{\prime} \boldsymbol{x}_{k}\right]
\end{equation}
$$

Deformable如下，其中的改进是使用了较小的数量$K$作为需要关注的区域，同时为同一个reference position预测了偏置，使其注意到周围的特征。
$$
\begin{equation}
\operatorname{DeformAttn}\left(\boldsymbol{z}_{q}, \boldsymbol{p}_{q}, \boldsymbol{x}\right)=\sum_{m=1}^{M} \boldsymbol{W}_{m}\left[\sum_{k=1}^{K} A_{m q k} \cdot \boldsymbol{W}_{m}^{\prime} \boldsymbol{x}\left(\boldsymbol{p}_{q}+\Delta \boldsymbol{p}_{m q k}\right)\right]
\end{equation}
$$

通过使用上述结构代替DETR中的对应自注意力模块，可以使得整个网络中，编码器的计算复杂度从$O(N^4)$降低为$O(N^2)$，解码器的计算复杂度从$O(NHWC)$变为$O(NKC)$，推理速度和训练速度都得以大幅提升。

作者同时引入Multi-scale的信息聚合，相当于为attention添加了一个维度。其同时对多个特征图上的特征进行聚合和加权。每个feature上使用不同的偏置进行学习。

当聚合的scale数量为1，关注的邻域位置数也为1，权重矩阵固定时，Deformable attention退化为deformable conv。

最后，相比DETR，因为这里只提取reference point周围的特征，因此这里对于目标框的预测是相对于reference points的。而不是DETR中的直接回归全局归一化坐标。另外，使用Proposal来作为解码器中的query。

最后，实验表明，其在性能、推理速度和训练速度上相比DETR均有明显的优势。
![](/assets/img/20220711/DeformableDETRF3.jpg)