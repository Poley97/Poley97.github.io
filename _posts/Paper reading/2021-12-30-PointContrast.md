---
layout: post
title: 'PointContrast: Unsupervised Pre-training for
3D Point Cloud Understanding'
date: 2021-12-30
author: Poley
cover: '/assets/img/20211228/PointContrast.png'
tags: 论文阅读
---

无监督（自监督）在3D上的工作开展的很少，作者认为主要由于三个原因：
1. 缺少大规模高质量数据
2. 缺少统一的Backbone
3. 缺少一个复杂的数据集的集合，以及高等级的任务用于评估。

无监督学习涵盖四项：
1. 找到大规模数据集用于预训练
2. 确定一个可以用于多个tasks的backbone
3. 在两个无监督目标上对于预训练的backbone进行评估
4. 定义一些下游任务来进行评估

通过实验，作者提出，用于预训练的数据，规模比标签的精度更重要。

本文提出两种不同的对比损失用于进行点云的预训练，分别是Hardest-contrastive loss和PointInfoNCE。

首先，无监督学习首先需要一个通用的Backbone，点云在这方面并不像2D那么统一。这是由于点云的无序性，和非结构化，导致了可以有多重不同的aggregation（近邻）方法用于处理。

## PointContrast Pre-tarining

之前的众多预训练方法在关注于ShapeNet模型，而这里作者发现在ShapeNet上的预训练反而会成为模型在下游任务上学习的阻碍，即出现了性能的下降，如下图所示。

![](/assets/img/20211228/PointContrastF1.png)

作者分析，这是由于两个原因造成：
+ Domain gap between source and target data: ShapeNet数据集中的数据都是合成，归一化，姿态对齐的，并且缺乏场景的上下文信息，因此和下游任务的分布区别较大。
+ Point-level representation matter: 3D任务重，局部的几何信息对网络的性能起到非常关键的作用。

因此，要寻找合适的预训练数据集，应该要满足两个条件
+ 具有复杂的场景和多物体
+ 网络结构不应该只关注与instance-level或global representation，而需要关注到 dense/local features。

这里使用的是FCGF的基本结构，但是由于FCGF本身的设计是用于registration这种low level的任务，因此作者在其基础上做了修改，增加了网络的复杂程度，用来提升网络的学习能力。

作者在这里使用的是ScanNet，其是一个RGBD的室内场景数据集，以序列形式存储。因此每一帧的数据可以通过相机坐标和参数轻松的投影到一个坐标系内，实现点的匹配。对于输入，首先产生两个不同view的输入点云（从序列中选择两个相距不太远的两帧就可以，文中的间隔是25帧），建立两幅点云之间点的对应匹配关系。之后**基本思想就是最小化匹配点的距离，最大化非匹配点的距离**。为此，作者退出两种Loss设计：

## Contrastive Learning Loss
+ Hardest-Contrastive Loss
$$
\begin{equation}
\mathcal{L}_{c}=\sum_{(i, j) \in \mathcal{P}}\left\{\left[d\left(\mathbf{f}_{i}, \mathbf{f}_{j}\right)-m_{p}\right]_{+}^{2} /|\mathcal{P}|+0.5\left[m_{n}-\min _{k \in \mathcal{N}} d\left(\mathbf{f}_{i}, \mathbf{f}_{k}\right)\right]_{+}^{2} /\left|\mathcal{N}_{i}\right|+0.5\left[m_{n}-\min _{k \in \mathcal{N}} d\left(\mathbf{f}_{j}, \mathbf{f}_{k}\right)\right]_{+}^{2} /\left|\mathcal{N}_{j}\right|\right\}
\end{equation}
$$

+ PointInfoNCE Loss
$$
\begin{equation}
\mathcal{L}_{c}=-\sum_{(i, j) \in \mathcal{P}} \log \frac{\exp \left(\mathbf{f}_{i} \cdot \mathbf{f}_{j} / \tau\right)}{\sum_{(\cdot, k) \in \mathcal{P}} \exp \left(\mathbf{f}_{i} \cdot \mathbf{f}_{k} / \tau\right)}
\end{equation}
$$

值得注意的是，InfoNCE Loss是从2D对比学习中演化过来的，但是其和2D对比学习的矛盾刚好相反。2D的对比损失考虑如何扩大负样本的数量，因此需要引入memory等方法来扩大对比的样本库，增加负样本数量；而3D的点太多了，这里需要控制负样本的数量。故只将有配对的点计入负样本。
从实验结果表明，后者的效果更好，在训练中体现出更加稳定，不容易崩溃的性质。