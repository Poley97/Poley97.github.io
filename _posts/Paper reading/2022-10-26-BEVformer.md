---
layout: post
title: 'BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers'
date: 2022-10-25
author: Poley
cover: '/assets/img/20221025/BEVFormer.jpg'
tags: 论文阅读
---

这篇文章提出使用Transformer实现multi-camera BEV特征在时序上的融合，从而实现更好、更鲁棒的计算性能。

目前来说，基于图像的BEV 3D检测性能不如其他的3D方法，因为从图像 to BEV本身是病态的，并且受到深度估计准确性的影响。因此，本文提出一种不依赖深度信息，并且可以自适应学习BEV特征而无需引入严格3D先验知识的方法。

本文使用网格的BRV queries用于融合空间和时序信息，通过注意力机制；空间的cross-attention来融合空间信息，从multi-camera images；时间上的自注意力，用于从历史bev特征中提取时序特征。大致概念图和具体结构图如下所示

![](/assets/img/20221025/BEVFormerF1.jpg)
![](/assets/img/20221025/BEVFormerF2.jpg)

相比传统的通过cat来堆叠多个BEV特征实现时序特征融合的方式，使用这样RNN-LIKE的方法（这里是使用transformer）在时间爱你窗口的操作上更加灵活，计算量也更小。

首先，多视图各自提取特征，得到图像feature map。之后，在bev平面上维护一个queries，其坐标系为自车坐标系，每一个query对应物理世界中的一个具体位置（位置通过可学习的位置编码引入，即坐标+MLP）。首先经过一个temporal self-attention 引入上一时刻的BEV特征。之后通过一个spatial cross-attention从相关的image feature map中提取特征，最终形成当前时刻的bev特征图。这个特征图后续可以被用于分割或者检测。具体细节如下

## Spatial Cross-Attention

这部分的功能主要是使用定义在BEV空间下的query来索引图像featrue map中的特征。由于BEV空间对应的物理位置和图像是有对应关系的，同时为了计算效率的问题，这里并不需要将每一个query和所有sensor feature map中的所有位置之间去做attention。

为了减小计算量，引入hit view的概念。即从BEV Query中对应的pillar中采样3D点，投影到图像中，寻找对应的hit view（该BEV出现在哪个图像里，以及在图像的什么位置）。根据这些reference point，去使用deformable attention来和参考点附近的local图像特征进行交互。并加权求和作为最终的输出。local特征的选择方式也是基于学习的，即使用deformable attention，如下所示

$$
\begin{equation}
\operatorname{SCA}\left(Q_p, F_t\right)=\frac{1}{\left|\mathcal{V}_{\text {hit }}\right|} \sum_{i \in \mathcal{V}_{\text {hit }}} \sum_{j=1}^{N_{\mathrm{mf}}} \operatorname{DeformAttn}\left(Q_p, \mathcal{P}(p, i, j), F_t^i\right)
\end{equation}
$$

但是，BEV每一个位置对应的是一个Pillar，而不是image中的唯一位置。这里实际上方式是在Pillar中的不同高度上取点（类似于LLS在每个Pixel的射线上取点）。空间采样点的z轴高度是预定义的。以捕捉不同高度上的信息。之后通过相机内参和外参矩阵投影到每一个图像上。而attention指处理query与其hit view之间的交互，而对于画面中不包含当前位置的图像则无需处理，可以避免无效的计算。

$$
\mathcal{P}(p, i, j)=\left(x_{i j}, y_{i j}\right)
$$
where $z_{i j} \cdot\left[\begin{array}{lll}x_{i j} & y_{i j} & 1\end{array}\right]^T=T_i \cdot\left[\begin{array}{llll}x^{\prime} & y^{\prime} & z_j^{\prime} & 1\end{array}\right]^T$.

## Temporal Self-Attention
时序信息通过当前时间的Q和上一时刻的BEV特征交互来获得。
1、对齐。根据上一时刻的自车运动将BEV对齐；
2、考虑到运动目标，BEV特征不可能严格对齐。因此引入deformAttn，来自适应的预测offset。
如下所示

$$
\begin{equation}
\operatorname{TSA}\left(Q_p,\left\{Q, B_{t-1}^{\prime}\right\}\right)=\sum_{V \in\left\{Q, B_{t-1}^{\prime}\right\}} \operatorname{DeformAttn}\left(Q_p, p, V\right)
\end{equation}
$$

通过这种方式，BEVFormer在纯相机方案上获得了很高的性能优势
![](/assets/img/20221025/BEVFormerT1T2.jpg)