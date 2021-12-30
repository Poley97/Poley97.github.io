---
layout: post
title: 'CornerNet: Detecting Objects as Paired Keypoints'
date: 2021-12-22
author: Poley
cover: '/assets/img/20211222/CornerNetF2.png'
tags: 论文阅读
---

CornerNet是一篇非常经典的文献，和传统的物体检测不同，其属于使用关键点检测来完成目标检测任务的框架。

本文的初衷在于解决传统的anchor-based方法的弊端:
+ Anchor需要很多，以及很多经验性的超参数
+ Anchor的正样本很少，存在正负样本的不平衡问题（主要指单阶段检测）

本文提出CornerNet，一种基于关键点的单接单目标检测器，通过检测目标框的两个角点（左上和右下）来实现目标检测。但是，如此检测存在一些问题——角点的判别并没有Local evidence来进行支撑，如下图所示。因此，传统的Pooling方法得到的local特征并不足以进行角点的判断。为此，本文提出一种新的pooling方法，corner pooling。

![](/assets/img/20211222/CornerNetF2.png)

网络的整体结构如下所示
![](/assets/img/20211222/CornerNetF4.png)

对于角点的检测，主要存在三个问题：
+ 正样本的划分
+ 角点的配对
+ 角点特征的提取

作者分别给出了对应的解决方案：

对于角点正样本的划分，很自然的把角点对应的heatmap上的位置作为正样本。但是如下图所示，在角点一定范围内的误差，仍然可以使得检测结果和gt具有很高的IoU，所以这不应该严重的惩罚。这里作者提出对角点附近的位置的损失适当降低。在角点附近确定一个半径，在这个半径中的角点可以使得检测框具有至少$t$的IoU(实际设置为0.5)，并在Heatmap上将其以高斯分布呈现。高斯分布的方差为半径的1/3。进而使用以下的改进版focal loss来弱化角点一定半径内的损失。
$$
\begin{equation}
L_{d e t}=\frac{-1}{N} \sum_{c=1}^{C} \sum_{i=1}^{H} \sum_{j=1}^{W}\left\{\begin{array}{cr}
\left(1-p_{c i j}\right)^{\alpha} \log \left(p_{c i j}\right) & \text { if } y_{c i j}=1 \\
\left(1-y_{c i j}\right)^{\beta}\left(p_{c i j}\right)^{\alpha} \log \left(1-p_{c i j}\right) & \text { otherwise }
\end{array}\right.
\end{equation}
$$

注意到将角点投影到heatmap上存在坐标的量化损失，本模型同样预测这个量化损失并将其恢复。Loss使用SmoothL1。

Grouping Corners同样是一个问题，因为当一个图形中有多个目标时，存在多组角点。这里作者参考了之前的经典工作，通过为每个角点预测一个embedding feature来进行配对。对于同一目标的角点，两者的embedding feature的距离应该尽可能接近，反之则尽可能远。故直接使用之前工作提出的Pull Push Loss来拉近匹配角点，推远不匹配角点，如下所示
$$
\begin{equation}
L_{p u l l}=\frac{1}{N} \sum_{k=1}^{N}\left[\left(e_{t_{k}}-e_{k}\right)^{2}+\left(e_{b_{k}}-e_{k}\right)^{2}\right]
\end{equation}
$$

$$
\begin{equation}
L_{p u s h}=\frac{1}{N(N-1)} \sum_{k=1}^{N} \sum_{j=1 \atop j \neq k}^{N} \max \left(0, \Delta-\left|e_{k}-e_{j}\right|\right),
\end{equation}
$$

在Corner pooling上，考虑到Corner预测的实际需求，即Corner的判断需要的不是自己的Local特征，而是自己位置沿两个方向上的特征，这里提出Corner pooling，即每个位置使用自己指定方向上的最显著特征（Max）作为自己的特征，如下所示。
$$
\begin{equation}
\begin{aligned}
t_{i j} &=\left\{\begin{array}{cc}
\max \left(f_{t_{i j}}, t_{(i+1) j}\right) & \text { if } i<H \\
f_{t_{H j}} & \text { otherwise }
\end{array}\right.\\
l_{i j} &=\left\{\begin{array}{cl}
\max \left(f_{l_{i j}}, l_{i(j+1)}\right) & \text { if } j<W \\
f_{l_{i W}} & \text { otherwise }
\end{array}\right.
\end{aligned}
\end{equation}
$$

通过动态规划，这在实现上变得非常简单，如下图所示。
![](/assets/img/20211222/CornerNetF6.png)

综上，就完成整个关键点的检测与匹配问题，最终实现目标检测任务。