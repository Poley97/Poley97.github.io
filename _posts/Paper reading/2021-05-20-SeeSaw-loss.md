---
layout: post
title: 'Seesaw Loss for Long-Tailed Instance Segmentation'
date: 2021-05-20
author: Poley
cover: '/assets/img/20210520/SeesawLoss.png'
tags: 论文阅读
---

> 论文链接 https://arxiv.org/abs/2008.10032
> 
> 参考博客 https://zhuanlan.zhihu.com/p/339126633

# 概述

COCO数据集一个均衡分布的目标检测数据集，已经提出了一系列方法来提高检测精度，而LVIS是一个具有常委腹部的目标检测/实力分割数据集。本文指出了指出了限制检测器在长尾分布数据上性能的一个关键原因：施加在尾部类别（tail class）上的**正负样本梯度的比例**是不均衡的。因此，作者提出**Seesaw Loss 来动态地抑制尾部类别上过量的负样本梯度，同时补充对误分类样本的惩罚**。 Seesaw Loss 显著提升了尾部类别的分类准确率，进而为检测器在长尾数据集上的整体性能带来可观的增益。

本文指出了尾部类别上的正负样本梯度的不平衡是影响长尾检测性能的关键因素之一。为了解决这个问题，作者提出了 Seesaw Loss 来针对性地调整施加在任意一个类别上的负样本梯度。给定一个尾部类别和一个相对更加高频的类别，**高频类施加在尾部类上的负样本梯度将根据两个类别在训练过程中累计样本数的比值进行减弱**。同时为了避免因负样本梯度减弱而增加的误分类的风险，Seesaw Loss 根据每个样本是否被误分类动态地补充负样本梯度。Seesaw Loss 有效地平衡了不同类别的正负样本梯度，提高了尾部类别的分类准确率，在长尾目标检测/实例分割数据集LVIS v1.0带来了上显著的性能提升。

在长尾分布的数据集中（例如：LVIS），大部分训练样本来自头部类别（head class），而只有少量样本来自尾部类别（tail class）。因此在训练过程中，来自头部类别的样本会对尾部类别施加过量的负样本梯度，淹没了来自尾部类别自身的正样本梯度。**这种不平衡的学习过程导致分类器倾向于给予尾部类别很低的响应，以降低训练的loss。**如下图所示，我们统计了在 LVIS v1.0 上训练Mask R-CNN过程中，施加在每个类别的分类器上正负样本累计梯度的分布。显然，头部类别获得的正负样本梯度比例接近1.0，而越是稀有的尾部类别，其获得的正负样本梯度的比例就越小。由此带来的结果就是分类的准确率随着样本数的减少而急剧下降，进而严重影响了检测器的性能。

![样本不平衡带来的正负梯度比例失衡](/assets/img/20210520/SeesawLossGradient.png)

#方法

为了方便直观理解，我们可以把正负样本梯度不均衡的问题，类比于一个一边放有较重物体而另一边放有较轻物体的跷跷板（Seesaw），如下图所示。为了平衡这个跷跷板，一个简单可行的方案就是缩短重物一侧跷跷板的臂长，即减少重物的重量在平衡过程中的权重。回到正负样本梯度不均衡的问题，我们提出了 Seesaw Loss 来动态地减少由头部类别施加在尾部类别上过量的负样本梯度的权重，从而达到正负样本梯度相对平衡的效果。

$$
\begin{equation}
\begin{array}{l}
L_{\text {seesaw }}(\mathbf{z})=-\sum_{i=1}^{C} y_{i} \log \left(\widehat{\sigma}_{i}\right)\\
\text { with } \widehat{\sigma}_{i}=\frac{e^{z_{i}}}{\sum_{j \neq i}^{C} \mathcal{S}_{i j} e^{z_{j}}+e^{z_{i}}}
\end{array}
\end{equation}
$$

$y \in \{0, 1\}$是one-hot label，$\bold{z}$是每一类预测的logit,由此可以得到其在每一类上的梯度

$$
\begin{equation}
\frac{\partial L_{\text {seesaw }}(\mathbf{z})}{\partial z_{j}}=\mathcal{S}_{i j} \frac{e^{z_{j}}}{e^{z_{i}}} \widehat{\sigma}_{i}
\end{equation}
$$

这样$\mathcal{S}_{ij}$就变为了一个系数，可以调整每一类上的样本梯度。通过选择合适的$\mathcal{S}_{ij}$，就可以达到平衡正负样本梯度的作用。

在 Seesaw Loss 的设计中，我们考虑了两方面的因素，一方面我们需要考虑类别间样本分布的关系（class-wise），并据此减少头部类别对尾部类别的"惩罚" （负样本梯度）；另一方面，盲目减少对尾部类别的惩罚会增加错误分类的风险，因为部分误分类的样本受到的惩罚变小了，因此对于那些在训练过程中误分类的样本我们需要保证其受到足够的"惩罚"。据此， $\mathcal{S}_{ij}$由两项相乘得到，
$$
\begin{equation}
\mathcal{S}_{i j}=\mathcal{M}_{i j} \cdot \mathcal{C}_{i j}
\end{equation}
$$

![](/assets/img/20210520/SeesawLossCurve.png)

## 【Mitigation Factor】
$\mathcal{S}_{ij}$的计算方法如下
$$
\begin{equation}
\mathcal{M}_{i j}=\left\{\begin{array}{cl}
1, & \text { if } N_{i} \leq N_{j} \\
\left(\frac{N_{j}}{N_{i}}\right)^{p}, & \text { if } N_{i}>N_{j}
\end{array}\right.
\end{equation}
$$

也就是说当第$i$类比第$j$类更加高频率地出现时，Seesaw Loss 就会自动根据两类之间不平衡的程度来减少第$i$类对第$j$类施加的负样本梯度。此外，我们在线地累计样本数量，而非使用预先统计的数据集样本分布，这样的设计主要是因为一些高级的样本 sampling 方式会改变数据集的分布（例如：repeat factor sampler, class balanced sampler 等）。在这种情况下，预先统计的方式无法反映训练过程中数据的真实分布。

## 【Compensation Factor】
为了防止过度减少负样本梯度而带来的分类错误，Seesaw Loss会增加对那些错误分类样本的惩罚。具体来说，如果一个第$i$ 类的样本错误分给了第$j$ 类，Seesaw Loss会根据两类之间的分类置信度的相对比值来适当的增加对第$j$类的惩罚。$\mathcal{C}_{ij}$的计算如下


## 【Normalized Linear Activation】
受到face recognition，few-shot learning等领域的启发，Seesaw Loss在预测分类logit的时候对weight和feature进行了归一化处理，即

$$
\begin{equation}
\begin{array}{c}
z=\tau \widetilde{\mathcal{W}}^{T} \tilde{x}+b \\
\widetilde{\mathcal{W}}_{i}=\frac{\mathcal{W}_{i}}{\left\|\mathcal{W}_{i}\right\|_{2}}, i \in C, \tilde{x}=\frac{x}{\|x\|_{2}}
\end{array}
\end{equation}
$$


## 【实验结果】

![](/assets/img/20210520/SeasawLossResult.png)

