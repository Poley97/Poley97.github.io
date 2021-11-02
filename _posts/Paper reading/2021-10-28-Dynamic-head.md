---
layout: post
title: 'Dynamic Head: Unifying Object Detection Heads with Attentions'
date: 2021-10-28
author: Poley
cover: '/assets/img/20211027/DynamicHead.png'
tags: 论文阅读
---

本文提出一种通用的object detection head with attention， 结合了三个主要问题： scale-awareness, spatial-awareness, task-awareness。


本文认为，作为一个好的通用的object detection head可以被总结为三个类别：
+ Scale-awareness，对于不同大小的物体都可以适应
+ Spatial-aware, 对不同形状，旋转和位置，视角的物体都可以适应
+ Task-aware， 对不同的任务都可以适应，比如目前的多种box的表示，box, center, corner points 和对应的回归任务。

本文将从backbone中提取的特征视为一个3d-tensor特征 $level \times space \times channel$，并将其视为一个attention learning problem.

## Related Work 

+ Scale-awareness: 多尺度特征的适应主要通过特征金字塔（FPN）来实现，但是其来自于Backbone的不同层，存在semantic gap。  

+ Spatial-awareness: 对于空间、旋转和多视角的适应主要通过增加模型容量、数据增强、deformable convolution等方法。

+ Task-awareness: 越来越多的工作发现使用物体的多种表征会提升模型的性能，包括使用Box和mask结合、center representation, ATSS等。

本工作融合了上面的三种特性。

# Approach

![](/assets/img/20211027/DynamicHeadF1.png)

作者将FPN中4D tensor表示为$\mathcal{F}\in\mathcal{R}^{L\times H \times W\times C}$，使用$S=H\times W$简化，特征变为$\mathcal{F}\in\mathcal{R}^{L\times S\times C}$，对应上述三个awareness。

其每个维度的作用如下：

+ $L$维度代表了不同level的特征，即scale。增强这个维度的特征有益于scale-awareness。
+ $S$维度代表了空间不同位置的特征，增强这个维度的特征有益于spatial awareness。
+ $C$这个维度代表了不同的表征和任务。增强它可以有益于task-awareness。

基于上述简单的思想，提出了Dynamic Head, 简单的说，就是对这三个维度做attention运算，可以表示为

$$
W(\mathcal{F})=\pi(\mathcal{F}) \cdot \mathcal{F}
$$

但是这样的计算量太大了，因此这里将其三个维度分别做Attention，个人感觉其实跟之前的空间和channel attention也没啥区别，只是多了一个scale。

$$
W(\mathcal{F})=\pi_{C}\left(\pi_{S}\left(\pi_{L}(\mathcal{F}) \cdot \mathcal{F}\right) \cdot \mathcal{F}\right) \cdot \mathcal{F}
$$

其中各部分分别如下：

+ Scale-aware Attention: 在另外两个维度上求和，并做一个简单的线性变换，通过1*1卷积实现。最后通过一个hard-sigmoid function 映射成一个权重。
$$
\pi_{L}(\mathcal{F}) \cdot \mathcal{F}=\sigma\left(f\left(\frac{1}{S C} \sum_{S, C} \mathcal{F}\right)\right) \cdot \mathcal{F}
$$

+ Spatial-aware Attention: 这里使用了deformable convolution，并在不同scale做平均。

$$
\pi_{S}(\mathcal{F}) \cdot \mathcal{F}=\frac{1}{L} \sum_{l=1}^{L} \sum_{k=1}^{K} w_{l, k} \cdot \mathcal{F}\left(l ; p_{k}+\Delta p_{k} ; c\right) \cdot \Delta m_{k}
$$

+ Task-aware Attention: 使用了两组参数，并通过max来动态的选择一组参数使用，如下
$$
\pi_{C}(\mathcal{F}) \cdot \mathcal{F}=\max \left(\alpha^{1}(\mathcal{F}) \cdot \mathcal{F}_{c}+\beta^{1}(\mathcal{F}), \alpha^{2}(\mathcal{F}) \cdot \mathcal{F}_{c}+\beta^{2}(\mathcal{F})\right)
$$

最终的结构，以及使用在不同结构网络上的示意图如下所示

![](/assets/img/20211027/DynamicHeadF2.png)

### Relation to Other Attention Mechanisms

作者把其他的Attention机制套到了自己这套泛化的框架里，还比较有意思。

+ Deformable： 相当于在 $S$,也就是空间上进行了attention。Deformable本质也就是对有限的空间位置进行学习的加权，可以看做attention。
+ Non-local: Non-local是一个早期工作，相当于在$L\times S$子空间内进行attention。即Scale和Spatial。
+ Transformer： 无需过多介绍。相当于在$S \times C$，空间和channel上进行attention。

## Experiments
本文对在上述几个子空间进行增强的效果进行了对比实验，如下所示。

![](/assets/img/20211027/DynamicHeadT1.png)

Attention的可视化如下

![](/assets/img/20211027/DynamicHeadF4.png)