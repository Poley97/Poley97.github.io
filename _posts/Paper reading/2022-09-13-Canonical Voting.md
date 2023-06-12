---
layout: post
title: 'Canonical Voting: Towards Robust Oriented Bounding Box Detection in 3D Scenes'
date: 2022-09-13
author: Poley
cover: '/assets/img/20220913/CV.jpg'
tags: 论文阅读
---

本文发表于CVPR2022，提出了一种新的适用于VoteNet的回归方式，以解决其offset和orientation预测不准确的问题。通过将offset分解成Local Canonical Coordinates的方法来实现。

VoteNet是目前3D检测的SOTA之一，其对每个点进行offset和orientation的预测，并通过聚类产生最终的Bounding box。但实际上，其对于offset和orientation的预测并不比随机猜测好多少，如下图所示

![](/assets/img/20220913/CVT1.jpg)

因此，本文提出将bounding box 的 regression分解，为局部坐标回归，并通过投票的方式来确定最终的方向和物体中心坐标。非常巧妙。总体结构如下图所示


![](/assets/img/20220913/CVF1.jpg)

## Regressing Local Canonical Coordinates

LCC是一个局部归一化坐标，其经过尺度缩放、旋转和平移之后可以转换到全局坐标。示意图如下图所示
![](/assets/img/20220913/CVF3.jpg)
$$
\begin{equation}
\begin{aligned}
\mathbf{p} &=\Psi_{\mathbf{s}, \alpha, \mathbf{t}}(\tilde{\mathbf{p}}) \\
&=\operatorname{diag}(\mathbf{s}) \cdot \mathbf{R}_y(\alpha) \cdot \tilde{\mathbf{p}}+\mathbf{t} \\
&=\left[\begin{array}{ccc}
s_x & 0 & 0 \\
0 & s_y & 0 \\
0 & 0 & s_z
\end{array}\right] \cdot\left[\begin{array}{ccc}
\cos (\alpha) & 0 & -\sin (\alpha) \\
0 & 1 & 0 \\
\sin (\alpha) & 0 & \cos (\alpha)
\end{array}\right] \cdot \tilde{\mathbf{p}}+\left[\begin{array}{c}
t_x \\
t_y \\
t_z
\end{array}\right],
\end{aligned}
\end{equation}
$$

实际上这个方法可以拓展到6自由度情况，因为对于局部坐标的回归均没有影响。之后，对每个点回归其尺度信息和局部坐标，损失如下
$$
L_{r e g}=\sum_{j=1}^{\left|\left\{B_j\right\}\right|} \sum_{i=1}^N\left(\left\|\mathbf{s}_j^*-\mathbf{s}_i\right\|+\left\|\Psi_{\mathbf{s}_j^*, \alpha_j^*, \mathbf{t}_j^*}^{-1}\left(\mathbf{p}_i\right)-\tilde{\mathbf{p}}_i\right\|\right)
$$
$\cdot \mathbb{1}\left(\mathbf{p}_i\right.$ on object $j$ 's surface $)$,

LCC相比与直接的偏移相比，其特点为具有旋转不变形。如下图所示，传统的offset和特征本身之间并没有显著的对应关系，同样的offset根据旋转的不同可以赋予物体的任何part，这种diversity并不利于网络的收敛
![](/assets/img/20220913/CVF4.jpg)

## Canonical Voting with Objectness

这一步作者通过投票来产生中心位置和方向，从而避免了直接的回归。如下图所示
![](/assets/img/20220913/CVF5.jpg)

对于每个点，由于已知其尺度和局部坐标，因此其对应的物体中心会出现在以其距离中心的距离为半径的的一个源上。这里作者将每个点的objectness赋给这个源上每个末端的体素（实际上是3*3的体素邻域，减小噪声）。相当与取多个圆的交点，因为这里只有局部坐标和尺度，因此不能确定准确的方向。但是可以通过多个圆的交点来确定物体中心。做个圆的交点由于被多次幅值，因此其会在heatmap上形成一个局部极大值，可以通过这一点来确定潜在的物体中心位置。

这个计算实际上是和通过投票实现的，即一个点对其周围一圈投票，上述操作可以通过GPU并行完成，时间复杂度O（N）。具体如下所示
![](/assets/img/20220913/CVA1.jpg)

## LCC Checking with Back Projection for Bounding Box Generation
由于上述实际上是对每个点可能的中心位置进行穷举，因此难免会出现FP，因为显然多个半径的交点并不止有物体中心处，并且在大量点的叠加下也不能保证极大值只出现在物体中心，如下图所示。

![](/assets/img/20220913/CVF6.jpg)

这里作者巧妙地结合之前的预测方法进行了反投影。即通过heatmap获得目标框的位置，方向和大小之后，使用这个候选框反过来框出其包含的点。由于每个点都有一个LLC局部预测，因此可以通过LLC局部预测与Heatmap中心预测是否consistent来判断这是否是一个明显的FP，非常巧妙，如下图所示

![](/assets/img/20220913/CVA2.jpg)

其中包含了两个阈值，首先是目标框中的Pos点数比例要足够多，其次是点和LCC预测中心和heatmap中心误差要足够小。这些在TP中都是理应满足的，因此可以用于筛选出Heatmap中错误的局部极大值引起的FP。

因为本文的方法中，所有前景点实际上都参与了对物体预测的投票，而没有使用聚类等方式，因此实际上对于被遮挡物体来说，相比VoteNet，本方法更加鲁棒。

但是，本方法似乎更加适合于应对单类别的情况。多个类别同时训练会导致性能的严重退化。如下图所示

![](/assets/img/20220913/CVT2.jpg)
