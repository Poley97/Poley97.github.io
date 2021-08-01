---
layout: post
title: 'Improving 3D Object Detection through Progressive Population Based Augmentation'
date: 2021-06-28
author: Poley
cover: '/assets/img/20210628/PP.png'
tags: 论文阅读
---

> 论文链接： https://arxiv.org/abs/2004.00831

本文提出**Progressive Population Based Augmentation(PPBA)**算法，通过不断缩小搜索空间，以及自适应的使用之前迭代发现的最优参数来**学习数据增强策略**。

# Introduction
在2D图像领域已经证明，在数据增强上大量投入，可能带来与更先进的模型带来的增益等量的增益。

主要贡献：
+ 提出一种新的对于3D点云的自动数据增强方法
+ 相比随机搜索数据增强，计算量更小，提升更明显
+ 可以提升10倍的数据效率
+ 同样可以泛化到2D情况

# Related wroks
其中介绍了一些2D、3D目标检测中常用的数据增强方法，略。

# Methods

作者将自动数据增强问题视为一个**hyperparameter schedule learning**的特殊情况。提出的方法主要包含两部分：
+ **一个特定的对点云输入的搜索空间**
+ **一个选择焕发来最优化数据增强参数**

## Search Space for 3D Point Cloud Augmentation

![](/assets/img/20210628/PPF1.png)
![](/assets/img/20210628/PPT7.png)
![](/assets/img/20210628/PPT8.png)

对于提出的搜索空间，数据增强策略包含N个增强操作。其中，每个操作都和一个概率以及若干超参数相联系。

对所提出的搜索空间包含的基本数据增强方法可以分为两类：**global operations and local operations**。所使用的方法如上图所示，一共8个操作，29个参数。

## Learning through Progressive Population Based Search

这个问题可以描述为，给定一个metric $\Omega$ 和模型 $\theta$ ，对于一组数据增强参数 $\lambda$ 寻找最优结果如下。其中这里的metric使用的是**mAP**。

$$
\begin{equation}
\boldsymbol{\lambda}^{*}=\arg \max _{\boldsymbol{\lambda} \in \Lambda^{T}} \Omega(\theta)
\end{equation}
$$

在训练过程中，一般使用objective function $L$来代替 metric $\Omega$，如下

$$
\begin{equation}
\boldsymbol{\theta}_{\boldsymbol{t}}^{*}=\arg \min _{\theta \in \Theta} L\left(\mathcal{X}, \mathcal{Y}, \lambda^{t}\right)
\end{equation}
$$

在搜索过程中，整个模型的训练被分为$N$次迭代，每次迭代，$M$的模型使用不同的数据增强参数$\lambda_t$来并行训练，并且使用metric $\Omega$ 来进行评测。所有在之前iterations中训练的模型都被放入 **population** $\mathcal{P}$。在初始的iteration，所有的模型和数据增强参数都是**randomly initialized**。

在第一个iteration结束之后，通过一个**exploit phase**来决定模型的参数。这个阶段从population $\mathcal{P}$中选择一个性能较好的 parent model。 之后进入 **exploration phase**，此时augmentation operation的一个子集会被explored，通过对其parent model中的对应增强参数进行**mutating(突变)**来做优化，同时剩下的增强参数直接从parent model继承。

和**Population Based Training**类似，**exploit phase**会保留好的模型，同时替换掉差的模型，在每个iteration结束之后。但是与其不同，本文提出的方法一次**只关注与数据增强超参数的一个子集**，而他的继任者可能会关心和前辈不同的另一个子集。因此，剩余的参数会根据目前最优的operations参数进行调整。对于还当前模型**还没有做过exploration的参数，在继承的时候，也继承对这个参数做过exploration的模型中性能最好的该参数的值，如下图所示。**

其流程如下所示
![](/assets/img/20210628/PPF2.png)

PPBA一次只调整搜索空间中的一小部分参数，并且之前迭代的历史信息在优化过程中也会被重复利用。通过将搜索范围缩小到当前的特定子集，这使得**区分差的数据增强参数变得更加容易**。为了减小因为单次搜索空间减小而造成的搜索速度下降，在每个iteration，每个seccessors的每个operation的最优参数都前辈中发现的最优参数中继承。

不同搜索范围策略的对比如下图所示，本文应该是第二种。

![](/assets/img/20210628/PPF3.png)