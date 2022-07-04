---
layout: post
title: '3D Object Detection with Pointformer'
date: 2021-12-30
author: Poley
cover: '/assets/img/20211228/Pointformer.png'
tags: 论文阅读
---

同样，Transformer相比CNN具有一定的优势，但是其复杂度是$O(n^2)$，这决定了不能直接使用全部点云作为输入。这里作者提出三个模块来在点云上应用Transformer，分别是Local Transformer， Local-Global Transformer 以及 Global Transformer。

本工作和VoTr可以分别作为点云检测网络可代替的Point Backbone和Voxel Backbone。

其整体结构如下图所示
![](/assets/img/20211228/PointformerF2.png)

目前常见的处理点的方法有两种，一种时使用简单的对称函数，比如Point-wise的前向网络或者pooling等。另一种是使用图网络，对近邻点的特征进行aggregating。前者不能有效的捕捉和利用点之间的上下文信息，而后者更多的关注与点之间的信息传递，而忽略了其相对关系。

## Local Transformer
如上所示，模块主要由三个子模块组成，其中Local Transformer是最主要的。其结构如下所示
![](/assets/img/20211228/PointformerF3.png)

这里使用的基本TransBlock如下所示
$$
\begin{equation}
\begin{gathered}
q_{i}^{(m)}=f_{i} W_{q}^{(m)}, k_{i}^{(m)}=f_{i} W_{k}^{(m)}, v_{i}^{(m)}=f_{i} W_{v}^{(m)}, \\
y_{i}^{(m)}=\sum_{j} \sigma\left(q_{i}^{(m)} k_{j}^{(m)} / \sqrt{d}+\operatorname{PE}\left(x_{i}, x_{j}\right)\right) v_{j}^{(m)}, \\
y_{i}=f_{i}+\operatorname{Concat}\left(y_{i}^{(0)}, y_{i}^{(1)}, \ldots, y_{i}^{(M-1)}\right), \\
o_{i}=y_{i}+\operatorname{FFN}\left(y_{i}\right),
\end{gathered}
\end{equation}
$$
这里的Local Transformer简单的来说，就是在PointNet++的Sample and Group的基础上，将一个Group视为一个Sequence，从而进行Transformer的操作，这样由于序列长度的大幅减少，复杂度会显著降低。在作者这种建模下，由于同样适用radius来寻找近邻，因此其和图神经网络也有相似之处。可以把DGCNN看做一种Local Transformer的特殊形式，因为其具有一部分固定设计的运算操作。通过Transformer，邻域点之间的相关性得以被考虑进来，而有时邻域点可以提供比中心点丰富的信息，这是其他模型中很少利用到的。

## Coordinate Refinement
FPS是最常用的点云下采样方法，可以在保持原有点云形状的同时进行下采样。本文同样使用这种方法对点云进行下采样。但是其具有两个缺点：
+ 对于outlier非常敏感；
+ 只能从已有点中进行采样，也就是当遇到遮挡或者视角问题等情况是，采样点也不能很好的描述物体的几何情况。

对此，这里对采样点坐标进行一个refinement，使其尽可能的接近目标的中心。具体的方法是使用Attetion map的特定行作为点坐标加权的权重，个人认为属于一种启发式的方法。如下所示。
$$
\begin{equation}
W=\frac{1}{M} \sum_{m=1}^{M} A_{0,:}^{(m)}
\end{equation}
$$
$$
\begin{equation}
x_{c_{t}}^{\prime}=\sum_{k=1}^{K} w_{k} x_{k}
\end{equation}
$$
## Global Transformer
这里类似PointNet++，有一个下采样和聚合的过程。在下采样之后进行一个Global Transformer，也类似PointNet++的全局特征，即将全部剩余的关键点视为一组，以相同的方法提取即可。

## Local-Global Transformer
这里是通过引入两组不同level的特征来做cross attention，原理也是一样的，只是key是来自于外部而不是自己的。如下所示，依然符合Tranblock的基本型形式。

$$
\begin{equation}
\begin{gathered}
f^{(0)}=\operatorname{FFN}\left(f_{i}\right), \forall i \in \mathcal{P}^{l}, \\
f_{j}^{\prime}=\operatorname{FFN}\left(f_{j}\right), \forall j \in \mathcal{P}^{h}, \\
F^{(l+1)}=\operatorname{Transblock}\left(F^{(l)}, F_{j}^{\prime}, \operatorname{PE}(X)\right), l=0, \ldots, L-1,
\end{gathered}
\end{equation}
$$

## Positional Encoding
这里在Group内将相对位移通过一个前向网络做位置编码，很简单，略。

## Computational Cost Reduction
这里参考了Linformer关于 Transformer的Low-rank简化的工作，并简单应用到了本文提出的网络中。实际上，确实也可以看做Group是一个低rank的，因为在实际操作中，为了方便程序实现，会为每个Group分配固定的大小。当Group没有那么多点时（大多数情况），会以0进行填充，此时就形成了一个low-rank的情况。因此这种近似对于点云情况带来的损失应该是相对较小的，是一种合理的简化方式。

## Experiments
![](/assets/img/20211228/PointformerT5T6.png)